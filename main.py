from Cen_DRL_Code import Cen_Env, Cen_train
from CompMADRL_Code import Comp_Env, Comp_MADRL_train
from CoopMADRL_Code import Coop_Env, Coop_MADRL_train
import Cen_DRL_Code.Cen_train as Cen
import CompMADRL_Code.Comp_MADRL_train as Comp
import CoopMADRL_Code.Coop_MADRL_train as Coop
import numpy as np

if __name__ == '__main__':
    epoch_max = 500
    # 跑不同SNR不同用户的
    Power_dB_arr = np.array([10, 20, 30])
    num_user_arr = np.array([2, 4, 6])
    max_K_arr = np.array([3])
    
    for num_user in num_user_arr:
        for Power_dB in Power_dB_arr:
            for max_K in max_K_arr:
                Cen.main(Power_dB, num_user, max_K, epoch=epoch_max)
                Comp.main(Power_dB, num_user, max_K, epoch=epoch_max)
                Coop.main(Power_dB, num_user, max_K, epoch=epoch_max)
                
    # 跑不同传输次数的
    Power_dB_arr = np.array([30])
    num_user_arr = np.array([6])
    max_K_arr = np.array([4,5,6,7,8,9,10])
    
    for num_user in num_user_arr:
        for Power_dB in Power_dB_arr:
            for max_K in max_K_arr:
                Cen.main(Power_dB, num_user, max_K, epoch=epoch_max)
                Comp.main(Power_dB, num_user, max_K, epoch=epoch_max)
                Coop.main(Power_dB, num_user, max_K, epoch=epoch_max)