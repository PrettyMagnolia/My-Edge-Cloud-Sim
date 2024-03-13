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
