import math

import pandas as pd
from matplotlib import pyplot as plt


def min_max_normalization(value, min_value, max_value):
    """
    归一化给定的值到[0, 1]区间内。

    :param value: 当前需要归一化的值
    :param min_value: 可能的最小值
    :param max_value: 可能的最大值
    :return: 归一化后的值
    """
    # 确保分母不为零
    if max_value - min_value == 0:
        return 0  # 或其他合适的默认值/处理
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value


def real_time_normalize(seq: list):
    if not seq or len(seq) < 2:
        # 序列为空或只有一个元素时，默认返回1
        return 1

    # 获取除最后一个元素外的序列部分
    sub_seq = seq[:-1]

    # 找到序列中的最大值和最小值
    max_value = max(sub_seq)
    min_value = min(sub_seq)

    # 防止除以零
    if max_value == min_value:
        return 0.5  # 如果所有值都相同，则归一化值设为0.5

    # 归一化最后一个值
    normalized_value = abs(seq[-1] - min_value) / (max_value - min_value)

    return normalized_value


def evaluate_policy(env, agent, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores / turns)


# You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise


if __name__ == '__main__':
    # 总奖励数据
    total_rewards = [
        -107.4198310338719, -41.928843860428586, -41.07653837198118, -42.22179004173107,
        -40.02154681041621, -40.56771331557179, -40.53301184613559, -58.18154914216829,
        -31.184183050979502, -32.20541352211993, -33.11843993983475, -35.54397621509267,
        -32.84702062243485, -26.690512351216572, -35.52076238966883, -31.61875474488695,
        -35.83964811317789, -35.4056235266234, -34.226608685508324, -33.027961904805906,
        -28.74921851632199, -30.82526507530396, -40.75565649538244, -34.19051636462381,
        -23.120918242455552, -29.506706432090436, -26.90707333984538, -30.764157312175115,
        -22.752695629684467, -24.398277181414752, -29.84925511582695, -28.734879162693417,
        -34.165098915667734, -26.312509871206508, -35.976668541624385, -27.98872066220297,
        -27.278544672153988, -39.636036229165484, -38.56772853055942, -28.123408251730265,
        -41.064088613607346, -25.09218767825366, -26.329652023900366, -32.17707342002128,
        -36.61454036012826, -31.427651013757885, -23.696716438045126, -26.688117522497922,
        -34.51189263465943, -26.260855481403752, -25.70464551305781, -28.24914670781852,
        -34.10411552773414, -39.027370158155755, -25.450364738184707, -33.66377745033559,
        -29.658744641497535, -26.267338074869404, -49.65519564535346, -23.73233086958566,
        -40.3089802654135, -25.967772933138985, -26.711088152243654, -25.284900772584265,
        -39.67443217086862, -28.35318893037314, -21.171391133958394, -34.013238009028704,
        -25.293022474088104, -26.716700944695866, -23.853748278292066, -25.09880586671101,
        -21.749115180332627, -27.452366548614247, -24.82944700952353, -40.471058404450346,
        -28.84294024146343, -25.876107972361797, -23.512572433645275, -27.81011433217264,
        -21.135835491132365, -21.48945052604395, -27.992223129464655, -37.829803997048415,
        -20.78082128471345, -22.10946967122008, -22.410721666194327, -32.1503700889361,
        -40.77810801928639, -22.838167504189094, -25.01951437676781, -34.884753731315904,
        -22.473117221519654, -33.40419090113886, -48.77009200082855, -23.35851526074746,
        -21.572923565063725, -19.347245936109196, -28.57865933771463, -17.817564973946748
    ]
    # 绘制折线图
    plt.figure(figsize=(14, 7))
    plt.plot(total_rewards, label='Total Reward per Episode', color='blue', marker='o')
    plt.title('Total Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()