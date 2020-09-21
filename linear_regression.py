import numpy as np
import matplotlib.pyplot as plt


"""
Script for playing around with different linear regression techniques 

"""

# create sample data
# simple f(x) = ax+b
# more general a and b are the weights w1 and w0: a = w1, b = w0
# the variable  x1 = x (slope) and x0 = 1 (offset) x0 is always 1
# --> f(x1,x0) = w1*x1 + w0*x0



def mse(y,y_predict):
        mse = np.mean((y-y_predict)**2)
        return mse

if __name__ == '__main__':
    print('hello')
    # In this case w1 and w0 have the following values
    w1 = 2
    w0 = 3
    print('w1:', w1)
    print('w0:',w0)

    # With w1 and w0 follows    f(x1,x0) = 2*x1 + 3*w0
    #                           f(x1,1)  = 2*x1 + 3*1

    # Vector notation w = [w1,w0] x = [x1,x0]
    # --> vector y_vec = w.T.dot(x) (scalar product)

    w_vec = np.array([w1,w0])


    # Examplary for a single point p1 with x1 and x0
    p1_x1 = 2
    p1_x0 = 1


    p1x = np.array([p1_x1,p1_x0])

    p1y = w_vec.dot(p1x.T)

    # print(p1x,p1_x1,p1y)

    # second point p2
    p2_x1 = 4
    p2_x0 = 1

    p2x = np.array([p2_x1,p2_x0])

    # combining the points to a sample vector
    x = np.array([p1x,p2x])

    # computing the y values
    y = w_vec.dot(x.T)
    y = w_vec.dot(x.T)
    print('x_vec:',x[:,0])
    print('y_vec:',y)

    # Creating n samples
    n = 100
    x = np.array(list([i,1] for i in range(n)))
    y = w_vec.dot(x.T)

    print('x_vec:',x[:,0])
    print('y_vec:',y)

    # plotting the undisturbed sample data (no noise)
    # plt.plot(x[:,0],y,linestyle = '',marker = '.',label = 'sample_no_noise')
    # plt.show()

    # Adding noise to y values
    t = 10*(np.random.random_sample((n,))-0.5)
    print(np.mean(t))
    yn = y+t

    # plotting the disturbed sample data (with noise)
    # plt.plot(x[:, 0], yn, linestyle='', marker='.',label = 'sample_with_noise')
    # plt.legend(loc = 0)
    # plt.xlabel('x1')
    # plt.ylabel('y')
    # plt.show()

    # Linear regression
    # optimize the parameters theta1 and theta0 in theta_vec so that they minimize the mean squared error
    # initial guess (random) of the parameters theta1 and theta0
    # theta1 = np.random.rand()
    # theta0 = np.random.rand()
    # theta_vec = [theta1,theta0]
    # print(theta_vec)
    # or
    theta_vec = np.random.random_sample((2,))
    print(theta_vec)

    y_predict = theta_vec.dot(x.T)

    # plotting the y_predict and yn values
    plt.plot(x[:, 0], y_predict, linestyle='-', marker='', label='y_predict')
    plt.plot(x[:, 0], yn, linestyle='', marker='.', label='sample_with_noise')
    plt.legend(loc=0)
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.show()

    # calculating the MSE
    error = mse(yn,y_predict)
    print('MSE of the initial guess:',error)

    # print(yn)