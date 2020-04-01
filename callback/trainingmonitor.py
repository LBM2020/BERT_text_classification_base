# encoding:utf-8
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ..common.tools import load_json
from ..common.tools import save_json
plt.switch_backend('agg')


class TrainingMonitor():
    def __init__(self, file_dir, arch, add_test=False):
        '''
        :param startAt: 重新开始训练的epoch点
        '''
        if isinstance(file_dir, Path):
            pass
        else:
            file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)

        self.arch = arch
        self.file_dir = file_dir
        self.H = {}
        self.add_test = add_test
        self.json_path = file_dir / (arch + "_training_monitor.json")

    def reset(self,start_at):
        if start_at > 0:
            if self.json_path is not None:
                if self.json_path.exists():
                    self.H = load_json(self.json_path)
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:start_at]

    def epoch_step(self, logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            # np.float32会报错
            if not isinstance(v, np.float):
                v = round(float(v), 4)
            l.append(v)
            self.H[k] = l

        # 写入文件
        if self.json_path is not None:
            save_json(data = self.H,file_path=self.json_path)

        # 保存train图像
        if len(self.H["loss"]) == 1:
            self.paths = {key: self.file_dir / (self.arch + f'_{key.upper()}') for key in self.H.keys()}

        if len(self.H["loss"]) > 1:
            # 指标变化
            # 曲线
            # 需要成对出现
            keys = [key for key, _ in self.H.items() if '_' not in key]
            for key in keys:
                N = np.arange(0, len(self.H[key]))
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, self.H[key], label=f"train_{key}")
                plt.plot(N, self.H[f"valid_{key}"], label=f"valid_{key}")
                if self.add_test:
                    plt.plot(N, self.H[f"test_{key}"], label=f"test_{key}")
                plt.legend()
                plt.xlabel("Epoch #")
                plt.ylabel(key)
                plt.title(f"Training {key} [Epoch {len(self.H[key])}]")
                plt.savefig(str(self.paths[key]))
                plt.close()
 class TrainLoss():
    def train_loss(self,steps,losses,epoch,args,type,max_step):
        plt.plot(steps,losses,'r-')#r代表red红色，-代表实线
        plt.xlabel('step')#设置xy轴的名称
        plt.ylabel('loss')
        plt.axis([0,max_step,0,1])#1）固定xy轴的刻度，避免每次生成的图因为刻度不一致不方便比较；2）xmin,xmax,ymin,ymax，xmax会随着数据量的变化而变化，此处直接作为参数直接传入，避免每次修改
        
        save_path = Path(f'{args.output_dir}/train_loss')
        if not save_path.exists():
            os.mkdir(save_path)
        if type == 'train':
            plt.title(f'loss_step in train_epoch{epoch}')
            plt.savefig(f'{save_path}/loss_step in train_epoch{epoch}.jpg')
        elif type == 'valid':
            plt.title(f'loss_step in valid_epoch{epoch}')
            plt.savefig(f'{save_path}/loss_step in valid_epoch{epoch}.jpg')
        else:
            raise ValueError
        plt.savefig(f'{save_path}/loss_step in epoch{epoch}.jpg')
        plt.close()#一定要close，否则会出现缓存，将第一张图片的内容打印在第二张图片上
        
