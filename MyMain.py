from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn as nn
from torch import optim
import MyClass
import MyData, pickle
import datetime
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 50
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5

import os, time, random
import argparse


## hyperparameters
parser = argparse.ArgumentParser(description='MT for Chinese to English')
parser.add_argument('--train_data', type=str, default='data', help='train data source')
parser.add_argument('--test_data', type=str, default='data', help='test data source')
parser.add_argument('--epoch_num', type=int, default=10, help='#epoch of training')
parser.add_argument('--hidden_size', type=int, default=256, help='#dim of hidden state')
parser.add_argument('--embedding_size', type=int, default=256, help='random init char embedding_dim')
parser.add_argument('--mode', type=str, default='test', help='train/test')
args = parser.parse_args()

import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

def tensorFromSentence(sent, vocab):
    #添加eos标记，并转换为tensor类型
    indexes = MyData.sentence2id(sent, vocab)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0], cn2id)
    target_tensor = tensorFromSentence(pair[1], en2id)
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    #encoder_outputs: [10,256]
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device) #decoder_input: [1,1]

    decoder_hidden = encoder_hidden[1,:,:].view(1,1,-1) #编码层最后一个时刻的隐藏状态作为解码层的初始状态

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False #随机确定是否使用teaching force

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1) #topk(n) : 求前n大的数（无序）
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, pairs, epoch_num, print_every=10, learning_rate=0.01):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(pair) for pair in pairs]
    n_iters = len(training_pairs)
    criterion = nn.NLLLoss()  # 负似然损失
    log = Logger('all.log', level='debug')

    for epoch in range(1, epoch_num+1):

        random.shuffle(training_pairs)
        print_loss_total = 0  # Reset every print_every

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            # 每次训练随机地选择是否使用teacher_forcing
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            # 每迭代若干次次输出一次平均损失
            if iter % print_every == 0:
                now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                #get_logger('log.txt').info('Epoch {}, {}/{}, loss: {:.5}'.format(epoch, iter, n_iters, print_loss_avg))

                string1 = 'Epoch: {}/{} iter: {}/{} loss: {:.4}' .format(epoch, epoch_num,
                                         iter, n_iters, print_loss_avg)
                log.logger.info(string1)

def evaluate(encoder, decoder, sentence, word2id, id2word, max_length=MAX_LENGTH):

    with torch.no_grad():
        input_tensor = tensorFromSentence(sentence, word2id)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden[1,:,:].view(1,1,-1)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(id2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateTest(encoder, decoder, pairs, vocab, id2tag):
    fw1 = open('result.txt','w', encoding='utf-8')
    total_score = 0
    for pair in pairs:
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], vocab, id2tag)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        total_score += sentence_bleu([pair[1]],output_words)
        s='> {}\nT: {}\nP: {}\n'.format(pair[0],pair[1],output_sentence)
        fw1.write(s)
    result_s="BLEU: {:.2}".format(total_score*1.0/len(pairs))
    print(result_s)
    fw1.close()


train_data_path = os.path.join('.', args.train_data, 'cn_train.txt')
train_label_path = os.path.join('.', args.train_data, 'en_train.txt')
test_data_path = os.path.join('.', args.test_data, 'cn_test.txt')
test_label_path = os.path.join('.', args.test_data, 'en_test.txt')
if not os.path.exists('train_data.pkl'):
    train_data = MyData.read_corpus(train_data_path, train_label_path, 'train')
    test_data = MyData.read_corpus(test_data_path, test_label_path, 'test')
    test_size = len(test_data)
else:
    print('loading existing data...')
    with open('train_data.pkl', 'rb') as fr:
        train_data = pickle.load(fr)
    with open('test_data.pkl', 'rb') as fr:
        test_data = pickle.load(fr)
        test_size = len(test_data)
vocab_path = os.path.join('.', args.train_data, 'word2id.pkl')
tag_path = os.path.join('.', args.train_data, 'tag2id.pkl')
id2tag_path = os.path.join('.', args.train_data, 'id2tag.pkl')
if not os.path.exists(vocab_path):
    MyData.vocab_build(vocab_path, tag_path, id2tag_path, train_data, 5)
cn2id, en2id, id2en = MyData.read_dictionary(vocab_path, tag_path, id2tag_path)


if args.mode == 'train':
    print('start training...')
    encoder1 = MyClass.EncoderRNN(len(cn2id), args.embedding_size, args.hidden_size).to(device)
    attn_decoder1 = MyClass.AttnDecoderRNN(args.hidden_size, len(en2id), dropout_p=0.1).to(device)
    trainIters(encoder1, attn_decoder1, train_data, args.epoch_num, print_every=64)  #75000：训练预料条数 5000：每5000次输出一次损失情况
    torch.save(encoder1,'model/encoder.pkl')
    torch.save(attn_decoder1,'model/decoder.pkl')

else:
    encoder = torch.load('model/encoder.pkl')
    decoder = torch.load('model/decoder.pkl')
    evaluateTest(encoder, decoder, test_data, cn2id, id2en)