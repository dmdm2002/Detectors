import pandas as pd
import os
from numba import cuda
from DNetPAD.Custom.train_DNetPAD_class_Ver import TrDNet
from AGPAD.train_PAM_CAM_Class_Ver import AGPAD
import time


class Controller(object):
    def __init__(self, Gmodels, Gmodels_infoDict):
        super(Controller, self).__init__()
        self.Gmodels = Gmodels
        self.Gmodels_infoDict = Gmodels_infoDict
        self.output = 'Z:/2nd_paper/backup/Compare/Detectors/Linux'
        os.makedirs(self.output, exist_ok=True)

    def _make_full_path(self, fold, path, Gmodel):
        if Gmodel != 'iDCGAN':
            if fold == '1-fold':
                tr_path = f'{path}/1-fold'
                te_path = f'{path}/1-fold'
            else:
                tr_path = f'{path}/2-fold'
                te_path = f'{path}/2-fold'
        else:
            tr_path = f'{path}/1-fold'
            te_path = f'{path}/1-fold'

        return tr_path, te_path

    def error_log(self, Detector, Gmodel, e):
        err_info = [Detector, Gmodel, e]
        # err = {'Detector': Detector, 'Gmodel': Gmodel, 'err': e}
        df = pd.DataFrame(err_info, columns=['Detector', 'GenerativeModel', 'err'])
        now = time.localtime()
        df.to_csv(f'{self.output}/{now.tm_year}-{now.tm_mon}-{now.tm_mday}_{now.tm_hour}-{now.tm_min}_err.csv')

    def start(self):
        history = []
        for Gmodel in Gmodels:
            if Gmodel == 'PGGAN' or Gmodel == 'CycleGAN':
                folds = ['1-fold']
            else:
                folds = ['1-fold', '2-fold']
            for fold in folds:
                path = Gmodels_infoDict[Gmodel]
                tr_path, te_path = self._make_full_path(fold, path, Gmodel)

                # DNetPAD
                try:
                    DNet = TrDNet(tr_path, te_path, Gmodel, fold)
                    best_score = DNet.run()
                    history.append(['DNet', Gmodel, fold, best_score['epoch'], best_score['acc'], best_score['apcer'], best_score['bpcer'], best_score['acer']])
                    df = pd.DataFrame(history, columns=['Detector', 'GenerativeModel', 'fold', 'epoch', 'acc', 'apcer', 'bpcer', 'acer'])
                    now = time.localtime()
                    df.to_csv(f'{self.output}/{now.tm_year}-{now.tm_mon}-{now.tm_mday}_{now.tm_hour}-{now.tm_min}_result.csv')

                except Exception as e:
                    self.error_log('DNet', Gmodel, e)

                device = cuda.get_current_device()
                device.reset()

                # try:
                #     ag = AGPAD(tr_path, te_path, Gmodel, fold)
                #     best_score = ag.run()
                #     history.append(['AGPAD', Gmodel, fold, best_score['epoch'], best_score['acc'], best_score['apcer'],
                #                     best_score['bpcer'], best_score['acer']])
                #     now = time.localtime()
                #     df = pd.DataFrame(history,
                #                       columns=['Detector', 'GenerativeModel', 'fold', 'epoch', 'acc', 'apcer', 'bpcer',
                #                                'acer'])
                #     df.to_csv(f'{self.output}/{now.tm_year}-{now.tm_mon}-{now.tm_mday}_{now.tm_hour}-{now.tm_min}_result.csv')
                #
                # except Exception as e:
                #     self.error_log('AGPAD', Gmodel, e)

                # device = cuda.get_current_device()
                # device.reset()


if __name__ == '__main__':
    Gmodels = ['UVC_GAN']
    Gmodels_infoDict = {
        'NestedUVC_DualAttention_Parallel_Fourier_MSE': 'Z:/2nd_paper/dataset/ND/ROI/Ablation/NestedUVC_DualAttention_Parallel_Fourier_MSE/',
        'StyTr2': 'Z:/2nd_paper/dataset/ND/ROI/Compare/StyTr2/',
        'UVC_GAN': 'Z:/2nd_paper/dataset/ND/ROI/Ablation/UVC_GAN/',
        'ACL-GAN': 'Z:/2nd_paper/dataset/ND/ROI/Compare/ACL-GAN/',
        'CycleGAN': 'Z:/2nd_paper/dataset/ND/ROI/Compare/CycleGAN/',
        'iDCGAN': 'Z:/2nd_paper/dataset/ND/ROI/Compare/iDCGAN/',
        'PGGAN': 'Z:/2nd_paper/dataset/ND/ROI/Compare/PGGAN/'
    }

    ct = Controller(Gmodels, Gmodels_infoDict)
    ct.start()
