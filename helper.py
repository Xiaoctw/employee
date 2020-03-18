import numpy as np
def cut_age(val):
    if val <= 25:
        return 0
    elif val <= 35:
        return 1
    elif val <= 55:
        return 2
    else:
        return 4


def cut_department(val):
    if val == 'Sales' or val == 'Human Resources':
        return 0
    return 1


def cut_dist(val):
    if val <= 10:
        return 0
    return 1


def cut_Edu(val):
    if val == 1:
        return 0
    elif val >= 5:
        return 1
    else:
        return 2


def cut_field(val):
    if val == 'Human Resources':
        return 0
    if val == 'Technical Degree':
        return 1
    if val == 'Marketing':
        return 2
    if val == 'Life Sciences' or val == 'Medical':
        return 3
    return 4


def cut_jobInvolve(val):
    if val == 1:
        return 0
    elif val == 4:
        return 2
    return 1


def cut_jobLevel(val):
    if val == 1:
        return 0
    elif val == 4:
        return 1
    else:
        return 2


def cut_JobRole(role):
    if role == 'Sales Representative':
        return 0
    elif role == 'Human Resources':
        return 1
    elif role == 'Laboratory Technician':
        return 2
    elif role == 'Research Scientist' or role == 'Sales Executive':
        return 3
    return 4


def cut_monthlyIncome(val):
    if val <= 3000:
        return 0
    elif val <= 6500:
        return 1
    else:
        return 2


def cut_percentSalaryHike(val):
    if val == 14 or val == 25:
        return 0
    elif val == 19 or val == 21:
        return 1
    elif val == 11 or val == 12 or val == 13 or val == 16 or val == 17 \
            or val == 18 or val == 20:
        return 2
    elif val == 22 or val == 23 or val == 15:
        return 3
    return 4


def cut_relationshipSatisfaction(val):
    if val == 1:
        return 0
    if val == 4:
        return 2
    return 1


def cut_totalWorkingYears(val):
    if val <= 5:
        return 0
    elif val <= 22:
        return 1
    elif val <= 35:
        return 2
    return 3


def cut_TrainingTimesLastYear(val):
    if val == 0:
        return 0
    elif val == 2 or val == 4:
        return 1
    return 2


def cut_WorkLifeBalance(val):
    if val == 1:
        return 0
    return 1


def cut_YearsAtCompany(val):
    if val <= 5:
        return 0
    elif val <= 10:
        return 1
    return 2


def cut_YearsInCurrentRole(val):
    if val <= 4:
        return 0
    elif val <= 8:
        return 1
    return 2


def cut_YearsSinceLastPromotion(val):
    if val <= 3:
        return 0
    elif val <= 8:
        return 1
    return 2


def cut_YearsWithCurrManager(val):
    if val <= 4:
        return 0
    elif val <= 10:
        return 1
    return 2

def find_deep_params(X):
    '''
    对X进行处理，得到可以进行嵌入矩阵
    :param X:
    :return:
    '''
    m,field_size=X.shape
    feat_sizes=[]
    dic,cnt={},0
    for i in range(field_size):
        feat_sizes.append(np.unique(X[:,i]).shape[0])
        list1=np.unique(X[:,i]).tolist()
        for val in list1:
            dic[i,val]=cnt
            cnt+=1
    for j in range(field_size):
        for i in range(m):
            val=X[i][j]
            X[i][j]=dic[j,val]
    return X,field_size,feat_sizes
