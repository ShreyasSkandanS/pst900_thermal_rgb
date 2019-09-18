close all;
clear;
clc;

model_dir = '/home/shreyas/model_testing/models/';
model_dir = strcat(model_dir,'dd_cleaner_labels_1/');

enc_train = strcat(model_dir,'encoder_train_log.txt');
enc_val   = strcat(model_dir,'encoder_val_log.txt');
dec_train = strcat(model_dir,'decoder_train_log.txt');
dec_val   = strcat(model_dir,'decoder_val_log.txt');

plot_ss_segmentation_data(enc_train);
plot_ss_segmentation_data(enc_val);
plot_ss_segmentation_data(dec_train);
plot_ss_segmentation_data(dec_val);

close all;
