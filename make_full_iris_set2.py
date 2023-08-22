from distutils.dir_util import copy_tree

FOLD_TYPE = ['1-fold']
checkpoint = ['280']
DB_NAME = 'NestedUVC_DualAttention_Parallel_Fourier_MSE'
DBs = ['A', 'B']


for i in range(len(FOLD_TYPE)):
    for db in DBs:
        print(f'--------------------------------NOW FOLD : [{FOLD_TYPE[i]}]--------------------------------')

        live = f'Z:/2nd_paper/dataset/ND/{db}_live'
        fake = f'Z:/2nd_paper/backup/GANs/ND/{DB_NAME}/{FOLD_TYPE[i]}/test/{db}/A2B/{checkpoint[i]}'

        copy_live = f'Z:/2nd_paper/dataset/ND/Full/Compare/{DB_NAME}/{FOLD_TYPE[i]}/{db}/live'
        copy_fake = f'Z:/2nd_paper/dataset/ND/Full/Compare/{DB_NAME}/{FOLD_TYPE[i]}/{db}/fake'

        copy_tree(fake, copy_fake)
        print(f'clear {db} fake')

        copy_tree(live, copy_live)
        print(f'clear {db} live')
