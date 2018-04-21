"""
Split data into train and test
"""
# coding:utf-8
import os
import sys
import shutil
curPath = os.path.abspath(os.path.dirname(__file__))
projectRootPath = curPath
sys.path.append(curPath)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
import argparse
import configparser
import re

parser = argparse.ArgumentParser()
parser.add_argument('--section', help='local, tencent or server', type=str)
args = parser.parse_args()
section = args.section

config = configparser.RawConfigParser()
config_path = projectRootPath + '/' + 'config.cfg'
config.read(config_path)
data_path = config.get(section, 'data_path')
test_ratio = config.getfloat(section, 'test_ratio')

def main():
    for i in range(1,3):
        logging.info("Split of dset{0} begins..."
                     .format(i))
        dset_path = data_path + '/dset%d' %i
        dset_path_train = dset_path + '/train'
        dset_path_test = dset_path + '/test'
        if not os.path.exists(dset_path_test):
            os.mkdir(dset_path_test)
        dset_label_dirs = os.listdir(dset_path_train)
        for label_dir in dset_label_dirs:
            label_dir_test = dset_path_test + '/' + label_dir
            if not os.path.exists(label_dir_test):
                os.mkdir(label_dir_test)
            count = 0
            files = os.listdir(dset_path_train + '/' + label_dir)
            for file in files:
                shutil.move(dset_path_train + '/' + label_dir+'/'+file,
                            dset_path_test + '/' + label_dir + '/' + file)
                count += 1
                if(count > test_ratio*len(files)):
                    break
        logging.info("Split of dset{0} has been finished."
                     .format(i))
if __name__ == '__main__':
    main()
