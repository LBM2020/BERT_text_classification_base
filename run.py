import os
import torch
import logging
import numpy as np
import argparse

from sklearn.metrics import classification_report

from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
from transformers.configuration_bert import BertConfig
from transformers import AdamW, WarmupLinearSchedule

from torch.utils.data import SequentialSampler,DataLoader

from process.dataprocess import processer
from process.progressbar import ProgressBar
from process.dataprocess import collate_fn
from process.rules import net_relus
from process.Adversarial_Training import FGM

from metrics.clue_compute_metrics import compute_metrics
from tools.common import seed_everything
from process.splitdata import split_data,get_output,write_pre_result_to_file

from callback.progressbar import ProgressBar
from callback.trainingmonitor import TrainLoss

from configs.config import config
logger = logging.getLogger()

def load_dataset(args,model_name_or_path,type):

    #仅用于定义变量
    input_file_name_or_path = ''
    max_seq_len = 0
    batch_size = 0

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    pro = processer()
    labellist = pro.get_labels()    

    if type == 'train':
        input_file_name_or_path = os.path.join(args.train_file_path,'train.txt')
        max_seq_len = args.train_max_seq_len
        batch_size = args.train_batch_size

    elif type == 'valid':
        input_file_name_or_path = os.path.join(args.valid_file_path,'valid.txt')
        max_seq_len = args.valid_max_seq_len
        batch_size = args.valid_batch_size

    elif type == 'test':
        input_file_name_or_path = os.path.join(args.predict_file_path,'test.txt')
        max_seq_len = args.predict_max_seq_len
        batch_size = args.predict_batch_size


    data = pro.read_txt(filename=input_file_name_or_path)
    examples = pro.create_examples(data=data, type=type)
    features = pro.convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                                      max_length=max_seq_len, label_list=labellist,
                                                      output_mode='classification')
    dataset = pro.create_dataset(features=features)

    sampler = SequentialSampler(dataset)#顺序取样
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size,
                                collate_fn=collate_fn)

    return data,dataloader

def train(args,model_name_or_path,train_data,train_dataloader,valid_data,valid_dataloader):

    pro = processer()
    labellist = pro.get_labels()
    trainloss = TrainLoss()

    #*****加载模型*****
    model = BertForSequenceClassification
    config = BertConfig.from_pretrained(model_name_or_path, num_labels=len(labellist))
    model = model.from_pretrained(model_name_or_path, config=config)

    # *****模型加载到设备*****
    if torch.cuda.is_available():
        # 单GPU计算
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)  # 设置GPU设备号
    else:
        device = torch.device('cpu')
    model.to(device)

    #*****优化函数*****
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(t_total * args.warmup_proportion)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)


    #*****训练过程相关信息*****
    print("***** Running training *****")
    print("  Num examples = %d", len(train_data))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  sequence length = %d", args.train_max_seq_len)
    print("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)

    #*****开始训练*****
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    seed_everything(args.seed)

    fgm = FGM(model)#创建对抗训练类的对象
    batches = []
    for batch in train_dataloader:
        batches.append(batch)

    train_steps = []
    train_losses = []  # 存储step和loss用于绘制loss曲线图
    for num in range(args.num_train_epochs):
        print(f'****************Train epoch-{num}****************')
        train_step = 0  # 代表一个epoch所有的步数
        global_step = 0
        pbar = ProgressBar(n_total=len(train_dataloader),desc='Train')

        # for step,batch in enumerate(train_dataloader):
        np.random.shuffle(batches)
        for step, batch in enumerate(batches):
            train_step += 1
            train_steps.append(train_step)#存储step用于绘制loss曲线

            model.train()

            #***输入模型进行计算***
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':batch[0],'attention_mask':batch[1],'token_type_ids':batch[2],'labels':batch[3]} #在Dataset中包含input_ids，attention_mask，token_type_ids，labels，len四个，在DataLoader中使用collate_fn函数在一个Batch中只添加了input_ids，attention_mask，token_type_ids，labels，没有添加len

            outputs = model(**inputs)  #模型原文件中已经使用损失函数对输出值和标签值进行了计算，返回的outputs中包含损失函数值

            #***损失函数值反向传播***
            loss = outputs[0]
            loss.backward()

            #***对抗训练***
            fgm.attack()
            loss_adv = model(**inputs)
            loss_adv = loss_adv[0]
            loss_adv.backward()
            fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)#梯度裁剪

            train_losses.append(loss.detach().cpu().numpy())#存储loss，绘制loss曲线

            #***优化器进行优化***
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()#优化器优化
                scheduler.step()#学习率机制更新
                model.zero_grad()
                global_step += 1

        #***一个epoch保存一个模型***
        output_dir = os.path.join(args.output_dir, f'model_checkpoint_epoch_{num}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('')  # 避免输出信息都在同一行
        logger.info(f'save model checkpoint-{global_step} to {output_dir} ')
        model.save_pretrained(output_dir)  # 保存模型

        # ***训练一个epoch绘制一个Loss曲线***
        print('绘制训练数据loss曲线')
        trainloss.train_loss(steps=train_steps, losses=train_losses, epoch=num, args=args,type='train')

        #*****一个epoch训练结束以后，进行验证*****
        print('')
        print(f'****************Valid epoch-{num}****************')
        print("  Num examples = %d", len(valid_data))
        print("  Batch size = %d", args.valid_batch_size)
        print("  sequence length = %d", args.valid_max_seq_len)
        valid_steps,valid_losses,valid_all_steps = valid(model=model,device=device,valid_dataloader=valid_dataloader)

        # ***验证一个epoch绘制一个验证集Loss曲线***
        print('绘制验证数据loss曲线')
        trainloss.train_loss(steps=valid_steps, losses=valid_losses, epoch=num, args=args, type='valid',max_step = valid_all_steps)

         #每训练一个epoch清空cuda缓存
        if 'cuda' in str(device):
            torch.cuda.empty_cache()


def valid(model,device,valid_dataloader):

    #*****开始验证*****
    preds = None
    # out_label_ids = None
    pbar = ProgressBar(n_total=len(valid_dataloader), desc="Evaluating")

    labels = []

    valid_step = 0  # 代表一个epoch所有的步数
    valid_steps = []
    valid_losses = []  # 存储step和loss用于绘制loss曲线图

    for step,batch in enumerate(valid_dataloader):
        #***存放step，绘制曲线图***
        valid_step += 1
        valid_steps.append(valid_step)  #

        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            # ***数据输入到模型中进行计算***
            inputs = {'input_ids':batch[0],'attention_mask':batch[1],'token_type_ids':batch[2],'labels':batch[3]}
            outputs = model(**inputs)

            #***计算损失***
            tmp_eval_loss, logits = outputs[:2]#1）tmp_eval_loss是损失函数值。2）logits是模型对验证集的预测概率值，例如二分类时,logits = [0.4,0.6]

            labels.extend(inputs['labels'].detach().cpu().numpy())#获取每个batch的真实标签，用于计算混淆矩阵

            #***存放loss，绘制曲线图***
            loss = outputs[0]
            valid_losses.append(loss.detach().cpu().numpy())

        if preds is None:
            #第一个batch时，preds为空
            preds = logits.softmax(-1).detach().cpu().numpy()
            # out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            #自第二个batch开始，将preds进行追加，例如第一个batch的preds为[[0.4,0.6],[0.3,0.7]],第二个batch的preds为[[0.2,0.8],[0.6,0.4]]
            #则追加（np.append）以后preds的值为[[0.4,0.6],[0.3,0.7],[0.2,0.8],[0.6,0.4]],其中每一个子list代表一个样本分属两个类别的概率
            #最后使用np.argmax对追加后的整个preds进行所有样本的类别判断
            preds = np.append(preds, logits.softmax(-1).detach().cpu().numpy(), axis=0)
            # out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        pbar(step)

    if 'cuda' in str(device):
        torch.cuda.empty_cache()


    #***输出该epoch验证集的混淆矩阵***
    true_label = labels
    pred_label = np.argmax(preds, axis=1)
    target_names = ['非涉网案件','涉网案件']
    print('')
    print(classification_report(y_true=true_label, y_pred=pred_label, target_names=target_names))
    print('')#避免输出信息都在同一行

    return valid_steps,valid_losses,valid_step

def predict(predict_model_name_or_path,pre_data,pre_dataloader):

    print('进行预测')
    pro = processer()
    labellist = pro.get_labels()

    #*****加载模型*****
    print('加载模型')
    model = BertForSequenceClassification
    config = BertConfig.from_pretrained(predict_model_name_or_path,num_labels = len(labellist))
    model = model.from_pretrained(predict_model_name_or_path,config=config)


    print('模型加载到GPU或者CPU')
    #如果有GPU，使用GPU进行分布式计算，否则使用CPU
    if torch.cuda.is_available():
        #单GPU计算
        torch.cuda.set_device(0)
        device = torch.device('cuda',0)#设置GPU设备号
    else:
        device = torch.device('cpu')
    model.to(device)

    print('******** Running prediction ********')
    print("  Num examples = %d", len(pre_data))

    preds = None
    pbar = ProgressBar(n_total=len(pre_dataloader), desc="Predicting")

    #***进行预测***
    for step, batch in enumerate(pre_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'token_type_ids' : batch[2],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            outputs = model(**inputs)
            _, logits = outputs[:2]

        #***汇总每个batch的预测结果***
        if preds is None:
            preds = logits.softmax(-1).detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.softmax(-1).detach().cpu().numpy(), axis=0)
        pbar(step)

    predict_label = np.argmax(preds, axis=1)
    print(preds)

    print(predict_label)
    return preds,predict_label

def run(args):

    seed_everything(args.seed)
    if args.do_train == 1:

        # print('划分训练数据和验证数据')
        # split_data(args=args, file_name_or_path=args.file_name_or_path)

        print('加载训练数据和验证数据')
        train_data, train_dataloader = load_dataset(args=args,model_name_or_path=args.model_name_or_path,type='train')
        valid_data, valid_dataloader = load_dataset(args=args, model_name_or_path=args.model_name_or_path, type='valid')
        print('训练数据和验证数据加载完成')

        print('开始训练')
        train(args=args,model_name_or_path=args.model_name_or_path,
              train_data=train_data,train_dataloader=train_dataloader,
              valid_data=valid_data,valid_dataloader=valid_dataloader)
        print('训练结束')

    if args.do_predict == 1:

        print('加载测试数据')
        pre_data,predict_dataloader = load_dataset(args=args,model_name_or_path=args.model_name_or_path,type='test')
        print('测试数据加载完成')

        print('开始预测')
        preds,predict_label = predict(predict_model_name_or_path=args.predict_model_name_or_path,
                                      pre_data=pre_data,pre_dataloader=predict_dataloader)
        print('预测完成')

        print('形成输出结果形式')
        target_names = ['非涉网案件', '涉网案件']
        outputlines = get_output(preds=preds,predict_label=predict_label,pre_data=pre_data,target_names=target_names)

        print('进行规则筛选')
        relu = net_relus()
        predict_result = relu.net_relus(nonet_keyword_list=config['nonet_keyword_list'],predict_result=outputlines)

        print('将预测结果写入文件')
        # predict_result = outputlines
        write_pre_result_to_file(args=args,output_lines=predict_result)
        print('预测结果写入完成')


def main():
    parser = argparse.ArgumentParser()
    #***需要在sh文件中定义的参数***
    parser.add_argument("--do_train",default=None,type=int,required=True,
                        help="Whether to run training.")
    parser.add_argument("--do_valid", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",default=None,type=int,required=True,
                        help="Whether to run the model in inference mode on the test set.")

    parser.add_argument("--file_name_or_path", default=None, type=str, required=True,
                        help="The input file direct,we will split the input file into train file and valid file.")
    parser.add_argument("--train_file_path", default=None, type=str, required=True,
                        help="The train file direct after split the input file")
    parser.add_argument("--valid_file_path", default=None, type=str, required=True,
                        help="The valid file direct after split the input file.")
    parser.add_argument("--predict_file_path", default=None, type=str, required=True,
                        help="The predict file direct after split the input file.")

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="The path of base model.")
    parser.add_argument("--predict_model_name_or_path", default=None, type=str, required=True,
                        help="The path of the model which is used to predict.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The path of output.")

    #***在py文件中定义的参数***
    parser.add_argument("--train_max_seq_len", default=256, type=str,
                        help="The max sequence length of train data.")
    parser.add_argument("--valid_max_seq_len", default=256, type=str,
                        help="The max sequence length of valid data.")
    parser.add_argument("--predict_max_seq_len", default=256, type=str,
                        help="The max sequence length of predict data.")

    parser.add_argument("--train_batch_size", default=8, type=str,
                        help="The batch_size of train data.")
    parser.add_argument("--valid_batch_size", default=8, type=str,
                        help="The batch_size of valid data.")
    parser.add_argument("--predict_batch_size", default=8, type=str,
                        help="The batch_size of predict data.")



    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--valid_size', type=int, default=0.1,
                        help="random seed for initialization")

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
