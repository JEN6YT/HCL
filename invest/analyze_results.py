import pickle, numpy, os

f = open('/home/ubuntu/code/HCL/invest/data/mar31_alleval_v4/sorted_v4_pkls.txt', 'r')
#f = open('/home/ubuntu/code/HCL/invest/data/mar31_alleval_v3/sorted_v3_pkls.txt', 'r')
#f = open('/home/ubuntu/code/HCL/invest/data/mar31_alleval_v2/sorted_v2_pkls.txt', 'r')
#f = open('/home/ubuntu/code/HCL/invest/data/mar31_alleval_v1/sorted_v1_pkls.txt', 'r')
l = f.readline()
eval_actual_returns = []

while l:
    print(l)
    D = pickle.load(open('/home/ubuntu/code/HCL/invest/data/mar31_alleval_v4/'+l.strip(), 'rb'))
    eval_actual_returns.append(D['eval_actual_return'][-1])
    l = f.readline()

s = 0
cnt = 0
for x in eval_actual_returns:
    if not numpy.isnan(x):
        s+=x
        cnt+=1
mean_return = s / cnt
sum_return = s

print('mean_return')
print(mean_return)

print('sum_return')
print(sum_return)

money = 100000

for x in eval_actual_returns:
    if not numpy.isnan(x):
        money = money * (1 + x)
print('starting money: '+str(100000))
print('resulting money: '+str(money))

print(eval_actual_returns)
