import socket
import sys
import numpy as np
from RC_wifi import RC
import matplotlib.pyplot as plt
from scipy.io import savemat
from tanh import tanh
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':


# ------------------------- LOS_Near --------------------#
#     package_num = 2 #27
#     folder_name = 'FPGA_new/LOS_Near/packet'
#     for i in range(package_num):
#         #x = np.linspace(0, 28800-1, 28800)
#         x = np.linspace(0, 99, 100)
#         test_in = np.fromfile(folder_name + str(i) + '/inputs.dat', sep='\n')
#         golden_in = np.fromfile('fpga/inputs.dat', sep='\n')
#         plt.figure(figsize=(8, 4))
#         plt.plot(x,test_in[0:100], label='$\sin x+1$', color='red', linewidth=2)
#         plt.plot(x, golden_in[0:100], label='$\sin x+1$', color='blue', linewidth=2)
#         plt.title('LOS_Near_input_packet_num= '+ str(i))
#         plt.show()
#
#         test_win = np.fromfile(folder_name + str(i) + '/w_in.dat', sep='\n')
#         golden_win = np.fromfile('fpga/w_in.dat', sep='\n')
#         x = np.linspace(0, 63, 64)
#         plt.figure(figsize=(8, 4))
#         plt.plot(x,test_win[0:64], label='$\sin x+1$', color='red', linewidth=2)
#         plt.plot(x, golden_win[0:64], label='$\sin x+1$', color='blue', linewidth=2)
#         plt.title('LOS_Near_win_packet_num= ' + str(i))
#         plt.show()
#
#         test_wout = np.fromfile(folder_name + str(i) + '/w_out.dat', sep='\n')
#         golden_wout = np.fromfile('fpga/w_out.dat', sep='\n')
#         x = np.linspace(0, 39, 40)
#         plt.figure(figsize=(8, 4))
#         plt.plot(x,test_win[0:40], label='$\sin x+1$', color='red', linewidth=2)
#         plt.plot(x, golden_win[0:40], label='$\sin x+1$', color='blue', linewidth=2)
#         plt.title('LOS_Near_wout_packet_num= ' + str(i))
#         plt.show()
#
#         test_predict = np.fromfile(folder_name + str(i) + '/predict_tosave.dat', sep='\n')
#         golden_predict = np.fromfile('fpga/predict_tosave.dat', sep='\n')
#         x = np.linspace(0, 99, 100)
#         plt.figure(figsize=(8, 4))
#         plt.plot(x,test_predict[0:100], label='$\sin x+1$', color='red', linewidth=2)
#         plt.plot(x, golden_predict[0:100], label='$\sin x+1$', color='blue', linewidth=2)
#         plt.title('LOS_Near_predict_packet_num= ' + str(i))
#         plt.show()




#------------------------- LOS_Far --------------------#
    # package_num1 = 22  # 22
    # folder_name1 = 'FPGA_new/LOS_Far/packet'
    # for i in range(package_num1):
    #     #x = np.linspace(0, 28800-1, 28800)
    #     x = np.linspace(0, 99, 100)
    #     test_in = np.fromfile(folder_name1 + str(i) + '/inputs.dat', sep='\n')
    #     golden_in = np.fromfile('fpga/inputs.dat', sep='\n')
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(x,test_in[4:104], label='$\sin x+1$', color='red', linewidth=2)
    #     plt.plot(x, golden_in[0:100], label='$\sin x+1$', color='blue', linewidth=2)
    #     plt.title('LOS_Far_input_packet_num= '+ str(i))
    #     plt.show()
    #
    #     test_win = np.fromfile(folder_name1 + str(i) + '/w_in.dat', sep='\n')
    #     golden_win = np.fromfile('fpga/w_in.dat', sep='\n')
    #     x = np.linspace(0, 63, 64)
    #     # plt.figure(figsize=(8, 4))
    #     # plt.plot(x,test_win[0:64], label='$\sin x+1$', color='red', linewidth=2)
    #     # plt.plot(x, golden_win[0:64], label='$\sin x+1$', color='blue', linewidth=2)
    #     # plt.title('LOS_Far_win_packet_num= ' + str(i))
    #     # plt.show()
    #
    #     test_wout = np.fromfile(folder_name1 + str(i) + '/w_out.dat', sep='\n')
    #     golden_wout = np.fromfile('fpga/w_out.dat', sep='\n')
    #     x = np.linspace(0, 39, 40)
    #     plt.figure(figsize=(8, 4))
    #     # plt.plot(x,test_win[0:40], label='$\sin x+1$', color='red', linewidth=2)
    #     # plt.plot(x, golden_win[0:40], label='$\sin x+1$', color='blue', linewidth=2)
    #     # plt.title('LOS_Far_wout_packet_num= ' + str(i))
    #     # plt.show()
    #
    #     test_predict = np.fromfile(folder_name1 + str(i) + '/predict_tosave.dat', sep='\n')
    #     golden_predict = np.fromfile('fpga/predict_tosave.dat', sep='\n')
    #     x = np.linspace(0, 99, 100)
    #     # plt.figure(figsize=(8, 4))
    #     # plt.plot(x,test_predict[0:100], label='$\sin x+1$', color='red', linewidth=2)
    #     # plt.plot(x, golden_predict[0:100], label='$\sin x+1$', color='blue', linewidth=2)
    #     # plt.title('LOS_Far_predict_packet_num= ' + str(i))
    #     # plt.show()




#------------------------- NLOS --------------------#
    package_num2 = 16  # 21
    folder_name2 = 'HW_results/S3/'
    MSE_list = []
    error_sign_rate_list = []
    for i in range(package_num2):
        # #x = np.linspace(0, 28800-1, 28800)
        # x = np.linspace(0, 99, 100)
        # test_in = np.fromfile(folder_name2 + str(i) + '/inputs.dat', sep='\n')
        # golden_in = np.fromfile('fpga/inputs.dat', sep='\n')
        # # plt.figure(figsize=(8, 4))
        # # plt.plot(x,test_in[8:108], label='$\sin x+1$', color='red', linewidth=2)
        # # plt.plot(x, golden_in[0:100], label='$\sin x+1$', color='blue', linewidth=2)
        # # plt.title('NLOS_packet_num= '+ str(i))
        # # plt.show()
        #
        # test_win = np.fromfile(folder_name2 + str(i) + '/w_in.dat', sep='\n')
        # golden_win = np.fromfile('fpga/w_in.dat', sep='\n')
        # x = np.linspace(0, 63, 64)
        # # plt.figure(figsize=(8, 4))
        # # plt.plot(x,test_win[0:64], label='$\sin x+1$', color='red', linewidth=2)
        # # plt.plot(x, golden_win[0:64], label='$\sin x+1$', color='blue', linewidth=2)
        # # plt.title('NLOS_win_packet_num= ' + str(i))
        # # plt.show()
        #
        # test_wout = np.fromfile(folder_name2 + str(i) + '/w_out.dat', sep='\n')
        # golden_wout = np.fromfile('fpga/w_out.dat', sep='\n')
        # x = np.linspace(0, 39, 40)
        # # plt.figure(figsize=(8, 4))
        # # plt.plot(x,test_win[0:40], label='$\sin x+1$', color='red', linewidth=2)
        # # plt.plot(x, golden_win[0:40], label='$\sin x+1$', color='blue', linewidth=2)
        # # plt.title('NLOS_wout_packet_num= ' + str(i))
        # # plt.show()
        #
        # test_predict = np.fromfile(folder_name2 + str(i) + '/predict_tosave.dat', sep='\n')
        # golden_predict = np.fromfile('fpga/predict_tosave.dat', sep='\n')
        # x = np.linspace(0, 99, 100)
        # # plt.figure(figsize=(8, 4))
        # # plt.plot(x,test_predict[0:100], label='$\sin x+1$', color='red', linewidth=2)
        # # plt.plot(x, golden_predict[0:100], label='$\sin x+1$', color='blue', linewidth=2)
        # # plt.title('NLOS_predict_packet_num= ' + str(i))
        # # plt.show()

        yout_hw = np.fromfile(folder_name2 + str(i) + '/y_out_hw.dat', sep='\n')
        yout_sw = np.fromfile(folder_name2 + str(i) + '/predict_tosave.dat', sep='\n')
        #yout_sw = np.fromfile(folder_name2 + str(i) + '/y_out_hw_no_update.dat', sep='\n')
        x = np.linspace(0, 14399, 14400)
        error = mean_squared_error(yout_sw,yout_hw)
        MSE_list.append(error)
        print('packet', str(i),', MSE:', error)
        plt.figure(figsize=(8, 4))
        #plt.plot(x,yout_hw[0:14400], label='$\sin x+1$', color='red', linewidth=2)
        #plt.plot(x, yout_sw[0:14400], label='$\sin x+1$', color='blue', linewidth=2)
        #plt.plot(x, yout_sw[0:14400]-yout_hw[0:14400],color='green', linewidth=2)z Q`
        #plt.plot(x, (yout_sw[0:14400] - yout_hw[0:14400])/max(yout_sw), color='yellow', linewidth=2)
        for j in range(len(yout_sw)):
            quotient = abs(yout_sw[j] / yout_hw[j])
            if quotient > 20 or quotient < 0.05:
                yout_hw[j] = yout_sw[j]
        plt.plot(x, (yout_hw[0:14400] / yout_sw[0:14400]), color='black', linewidth=2)
        plt.title('predict_packet_num= ' + str(i))
        plt.show()
        #print(np.sign(yout_sw),np.sign(yout_hw))
        error_sign_cnt = 0
        for i in range(len(yout_sw)):
            if np.sign(yout_sw[i]) != np.sign(yout_hw[i]):
                error_sign_cnt = error_sign_cnt + 1
        print('error_sign_rate', error_sign_cnt/len(yout_sw))
        error_sign_rate_list.append(error_sign_cnt/len(yout_sw))
        #np.where((np.sign(yout_sw)+np.sign(yout_hw))==0)
    print('average_MSE:', sum(MSE_list)/len(MSE_list))
    print('average_err_sign_rate:', sum(error_sign_rate_list)/len(error_sign_rate_list))