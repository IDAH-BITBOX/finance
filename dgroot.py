import math
import numpy as np


def sim_package(data, predict, start, end):     # start 와 end 는 ensemble의 인덱스
    data = np.array(data)
    predict = np.array(predict)
    result = dict()


def dirac_function(x, data):
    for i in range(len(data)):
        if x != i:
            continue
        else:
            return data[i]


def data_reverse(data):
    n = len(data)
    data_ = list()
    for i in range(n):
        data_.append(data[n - i - 1])
    return data_


def discrete_to_continuous(data, x):
    n = len(data)
    y = 0
    point = 1 / x

    for k in range(1, n+1):
        c = 0
        for j in range(1, k+1):
            c += (math.factorial(k-1)/(math.factorial(j-1)*math.factorial(k-j))*pow(-1.0, k+j)*data[j-1])
        point = point*(x - k + 1)
        y += c/math.factorial(k-1)*point

    return y


def continuous_regression(data, n):
    error = 0.000000001
    value = discrete_to_continuous(data, n + error)
    value = (value - data[n - 1])/error
    return value


def trend_regression(data, n):
    error = 0.5
    value_1 = discrete_to_continuous(data, n + error)
    value_2 = discrete_to_continuous(data, n - error)
    value = (value_1 - value_2) / (2 * error)
    return value


def tangent(data_, start_index):
    _data = []
    tau = 0.1
    for i in range(start_index, len(data_) - 1):
        ab = pow(tau * tau + data_[i] * data_[i], 0.5) * pow(tau * tau + data_[i + 1] * data_[i + 1], 0.5)
        cos = (tau * tau + data_[i] * data_[i + 1]) / ab
        cos = int(cos * 1000000000000000) / 1000000000000000.0
        tan_product = ab * pow(1.0 - cos * cos, 0.5) / cos
        _data.append(tan_product)
    return _data


def tangent_judge(data_):
    judge = []
    for i in range(len(data_)):
        if data_[i] < 0.0:
            judge.append("둔각")
        elif data_[i] > 1000.0:
            judge.append("직각")
        else:
            judge.append("예각")
    return judge


def logistic_hypothesis(theta_list, x_list):
    z = theta_list[0]
    for i in range(1, min(len(x_list), len(theta_list))):
        z += theta_list[i] * x_list[i]
    print(z)
    return [1 / (1 + math.exp(z)), math.exp(z)]
# theta_list 는 0부터, x_list(예측한 함수)는 1부터 채워진다.


def logistic_cost(theta_list, x_list, y_list):
    result = 0.0
    result1 = 0.0
    for i in range(min(len(y_list), len(x_list), len(theta_list))):
        tmp = -y_list[i] * math.log(logistic_hypothesis(theta_list, x_list)[0]) - (1.0 - y_list[i]) * math.log(1.0 - logistic_hypothesis(theta_list, x_list)[0])
        result += tmp
        result1 += logistic_differentiate(theta_list, x_list, x_list[i], y_list[i])
    return [result / len(y_list), result1 / (-len(y_list))]


def logistic_differentiate(theta_list, x_list, x, y):
    return y * math.log(logistic_hypothesis(theta_list, x_list)[0]) * (math.log(logistic_hypothesis(theta_list, x_list)[1]) * (-x) * math.log(logistic_hypothesis(theta_list, x_list)[0])) + (1.0 - y) * math.log(1.0 - logistic_hypothesis(theta_list, x_list)[0]) * math.log(logistic_hypothesis(theta_list, x_list)[0]) * (-x)


def logistic_gradient(x_list, y_list, alpha, theta_length):
    theta_list = []
    temp_list = []
    for j in range(theta_length):
        theta_list.append(0.0)
    for i in range(min(len(y_list), len(x_list), len(theta_list))):
        temp_list.append(theta_list[i] - alpha * logistic_cost(theta_list, x_list, y_list)[1])

        theta_list[i] = temp_list[i]
    return theta_list


def hypothesis_function(theta_0, theta_1, x):
    return theta_0 + theta_1 * x


def cost_function_basis(theta_0, theta_1, x_i,  y_i, m):
    result = 1 / (2 * m) * pow((hypothesis_function(theta_0, theta_1, x_i) - y_i), 2.0)
    result_1 = 1 / m * (hypothesis_function(theta_0, theta_1, x_i) - y_i)
    result_2 = 1 / m * (hypothesis_function(theta_0, theta_1, x_i) - y_i) * x_i
    return [result, result_1, result_2]


def cost_function_del(x, y, theta_0, theta_1, m):   # x, y : list / theta_0, theta_1 : where the least cost
    result_0 = 0.0
    result_1 = 0.0
    for i in range(m):
        result_0 += cost_function_basis(theta_0, theta_1, x[i], y[i], m)[1]
        result_1 += cost_function_basis(theta_0, theta_1, x[i], y[i], m)[2]
    return [result_0, result_1]


def gradient_descent(x, y, alpha, m):
    theta_0 = 0
    theta_1 = 0
    for i in range(m):
        temp_0 = theta_0 - alpha * cost_function_del(x, y, theta_0, theta_1, m)[0]
        temp_1 = theta_1 - alpha * cost_function_del(x, y, theta_0, theta_1, m)[1]

        theta_0 = temp_0
        theta_1 = temp_1
    return [theta_0, theta_1]


def cost_function(x, y, alpha, m):   # x, y : list / theta_0, theta_1 : where the least cost
    result = 0.0
    theta_0 = gradient_descent(x, y, alpha, m)[0]
    theta_1 = gradient_descent(x, y, alpha, m)[1]
    for i in range(m):
        result += cost_function_basis(theta_0, theta_1, x[i], y[i], m)[0]
    return result


def difference(data_, start_index):
    _data = []
    for k in range(start_index):
        _data.append("")
    for i in range(len(data_) - 1):
        if data_[i] != '':
            _data.append(data_[i] - data_[i + 1])
        else:
            _data.append("")
    return _data


def sum_partial(data_, start_index, range_):
    _data = []
    for k in range(start_index):
        _data.append("")
    for i in range(len(data_) - range_):
        if data_[i] != '':
            _data_ = 0
            for j in range(range_):
                _data_ += data_[i + j]
            _data.append(_data_)
        else:
            _data.append("")
    return _data


def abs_data(data_, start_index):
    _data = []
    for k in range(start_index):
        _data.append("")

    for i in range(len(data_)):
        if data_[i] == '':
            _data.append('')
        else:
            _data.append(abs(data_[i]))

    return _data


def sign(data_, start_index):
    _data = []
    for k in range(start_index):
        _data.append("")
    for i in range(len(data_)):
        if data_[i] == 0:
            _data.append(0)
        elif data_[i] == '':
            _data.append('')
        else:
            _data.append(data_[i] / abs(data_[i]))
    return _data


def sub_partial(data_, start_index, range_):
    _data = []
    for k in range(start_index):
        _data.append("")
    for i in range(len(data_) - range_):
        if data_[i] != '':
            _data.append(data_[i] - data_[i + range_])
        else:
            _data.append("")
    return _data


def count_if(data, start_num, count_range, value):
    if type(data) == list:
        seq_ = data[start_num:start_num + count_range]
    else:
        seq_ = data.tolist()[start_num:start_num + count_range]

    return seq_.count(value)


def sum_data(data_1, data_2, data_gap, start_index):
    _data = []
    for k in range(start_index):
        _data.append("")
    for i in range(min(len(data_1), len(data_2)) - data_gap):
        if data_1[i] != '':
            _data.append(data_1[i] + data_2[i + data_gap])
        else:
            _data.append("")
    return _data


def sub_data(data_1, data_2, data_gap, start_index):
    _data = []
    for k in range(start_index):
        _data.append("")
    for i in range(min(len(data_1), len(data_2)) - data_gap):
        if data_1[i] != '' and data_2[i + data_gap] != '':
            _data.append(data_1[i] - data_2[i + data_gap])
        else:
            _data.append("")
    return _data


def difference_refine(data, delay, delay_const):
    data_ = list()
    for i in range(delay_const, len(data)):
        sub = data[i - delay] - data[i]
        data_.append(sub)
    return data_


def simulation(raw_data, refined_data, delay):
    refined_data_ = refined_data[delay:]
    check = list()
    for j in range(1, delay + 1):
        raw_data_ = difference_refine(raw_data, j, delay)
        check.append(raw_data_)
    # print(check)
    # print(refined_data_)
    count = 0
    k = 0
    for k, item in enumerate(refined_data_):
        count_minimum = 0
        for i in range(delay):
            if item * check[i][k] > 0:
                count_minimum = 1
        count += count_minimum
    count /= k+1
    return count


'''
resu = [1, 1, -1, 1, -1, 1, 1, -1, 1, -1]
diff = [5000, 3000, 5000, 4000, 6000, 4000, 3000, 5000, 1000, 2000]

print(simulation(diff, resu, 1))
'''

'''
resul = [1, 2, 3, 4, 5, 1, 2, 3, 2, 1]
price = [2, 3, 4, 5, 1, 3, 2, 3, 1, 1, 5, 4, 3, 4, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]

diff = difference(price, 1)
shape = difference(diff, 0)
shape_net = sum_partial(shape, 0, 5)
stand = sum_data(price, shape, 1, 0)
price_stand = sub_data(price, stand, 0, 0)[1:]

print(diff, shape, shape_net, stand, price_stand)


# 2차 데이터(부호화)
diff = sign(diff, 0)
shape = sign(shape, 0)
shape_net = sign(shape_net, 0)
price_stand = sign(price_stand, 0)

# 3차 데이터(합손질)
diff_shape_sum = sum_data(diff, shape, 0, 0)
shape_shape_net_sum = sum_data(shape, shape_net, 0, 0)
shape_sum = sum_partial(shape, 0, 2)
shape_net_sum = sum_partial(shape_net, 0, 2)
price_stand_sum = sum_partial(price_stand, 0, 2)

# 3차 데이터(유량 손질)
shape_flow = sum_partial(shape, 0, 5)
shape_net_flow = sum_partial(shape_net, 0, 5)
price_stand_flow = sum_partial(price_stand, 0, 5)

shape_flow = sum_partial(shape_flow, 0, 2)
shape_net_flow = sum_partial(shape_net_flow, 0, 2)
price_stand_flow = sum_partial(price_stand_flow, 0, 2)

result = []

for i in range(1, min(len(shape_shape_net_sum), len(shape_net_flow))):
    res = -(diff_shape_sum[i] + price_stand_sum[i] + shape_net_sum[i] + shape_sum[i] + shape_shape_net_sum[i] + price_stand_flow[i] + shape_flow[i] + shape_net_flow[i])
    if res == 0:
        result.append(0)
    else:
        result.append(abs(res)/res)

print(result)'''