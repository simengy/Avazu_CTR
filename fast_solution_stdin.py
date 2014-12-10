'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt


# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
train = 'train.csv'               # path to training file
test = 'test.csv'                 # path to testing file
submission = 'grid_search/result/submission_0.41_1.8_1.8.csv' # path of to be outputted submission file

# B, model
alpha = 0.41 # learning rate
beta = 1   # smoothing parameter for adaptive learning rate

L1 = 1.8 # L1 regularization, larger value means more regularized
L2 = 1.8 # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 30             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1        # learn training data for N passes

holdbefore = 24  # data before date N (exclusive) are hidden because of time-series
holdafter = None  # data after date N (exclusive) are used as validation
holdout = 10  # use every N training instance for holdout validation

dayfilter = None
dayfeature = True
counters = False

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D, dayfilter = None, dayfeature = True, counters = False):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''
    
    device_ip_counter = {}
    device_id_counter = {}

    for t, row in enumerate(DictReader(open(path))):
        # process id
        ID = row['id']
        del row['id']

        # process clicks
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        # extract date
        date = int(row['hour'][4:6])

        # turn hour really into hour, it was originally YYMMDDHH
        row['hour'] = row['hour'][6:]
        
        if row['C20'] != '-1':
            row['C20'] = str(int(row['C20']) - 100000)
        
        if dayfilter != None and not date in dayfilter:
            continue

        if dayfeature: 
            # extract date
            row['wd'] = str(int(date) % 7)
            row['wd_hour'] = "%s_%s" % (row['wd'], row['hour'])            

        if counters:
            d_ip = row['device_ip']
            d_id = row ["device_id"]
            try:
                device_ip_counter [d_ip] += 1
                device_id_counter [d_id] += 1
            except KeyError:
                device_ip_counter [d_ip] = 1
                device_id_counter [d_id] = 1                
            row["ipc"] = str(min(device_ip_counter[d_ip], 8))
            row["idc"] = str(min(device_id_counter[d_id], 8))

        # build x
        x = []
        for key in row:
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        # app, site, device interactions
	app_col = ['app_id', 'app_domain', 'app_category']
        site_col = ['site_id', 'site_domain', 'site_category']
        device_col = ['device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type']

        id_col = ['app_id', 'site_id', 'device_id']
        domain_col = ['app_domain', 'site_domain', 'device_model']
        category_col = ['app_category', 'site_category', 'device_type']
	
        size_col = ['C15', 'C16']
	connSize_col = ['C15', 'C16', 'device_conn_type']
        
        resolution_col = ['C20', 'C21']
	bannerRes_col = ['banner_pos', 'C20', 'C21']
        sizeRes_col = ['C15', 'C16', 'C20', 'C21']

	#app
	index = abs(hash( row[app_col[0]] + '_x_' + row[app_col[1]] + '_x_' + row[app_col[2]] )) % D
        x.append(index)
        #site
	index = abs(hash( row[site_col[0]] + '_x_' + row[site_col[1]] + '_x_' + row[site_col[2]] )) % D
        x.append(index)
	#device
	index = abs(hash( row[device_col[0]] + '_x_' + row[device_col[1]] + '_x_' + row[device_col[2]] + '_x_' + row[device_col[3]] + '_x_' + row[device_col[4]])) % D
        x.append(index)
        
	#id
	index = abs(hash( row[id_col[0]] + '_x_' + row[id_col[1]] + '_x_' + row[id_col[2]] )) % D
        x.append(index)
	#domain
	index = abs(hash( row[domain_col[0]] + '_x_' + row[domain_col[1]] + '_x_' + row[domain_col[2]] )) % D
        x.append(index)
	#category
	index = abs(hash( row[category_col[0]] + '_x_' + row[category_col[1]] + '_x_' + row[category_col[2]] )) % D
        x.append(index)
        '''
	#size
	index = abs(hash( row[size_col[0]] + '_x_' + row[size_col[1]] )) % D
        x.append(index)
        '''
        #connSize
	index = abs(hash( row[connSize_col[0]] + '_x_' + row[connSize_col[1]] + '_x_' + row[connSize_col[2]] )) % D
        x.append(index)
        
        #resolution
	index = abs(hash( row[resolution_col[0]] + '_x_' + row[resolution_col[1]] )) % D
        x.append(index)
        '''
        #bannerResSize
	index = abs(hash( row[bannerRes_col[0]] + '_x_' + row[bannerRes_col[1]] + '_x_' + row[bannerRes_col[2]] )) % D
        x.append(index)
        #sizeResSize
	index = abs(hash( row[sizeRes_col[0]] + '_x_' + row[sizeRes_col[1]] + '_x_' + row[sizeRes_col[2]] + '_x_' + row[sizeRes_col[3]] )) % D
        x.append(index)
        '''
        # pair-wise interactions
        '''
        all_col = app_col
        all_col.extend(site_col)
        all_col.extend(device_col)
        '''
        all_col = ['C14','C15','C16','C17','C18','C19','C20','C21']
        L = len(all_col)

        for i in range(L):
            for j in range(i+1, L):
                index = abs(hash( row[all_col[i]] + '_x_' + row[all_col[j]] )) % D
                x.append(index)
        
        yield t, date, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
p = 0
# start training
for e in xrange(epoch):
    loss = 0.
    count = 0

    for t, date, ID, x, y in data(train, D, dayfilter = dayfilter, dayfeature = dayfeature, counters = counters):  # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)

        # step 1, get prediction from learner
        if date != None:
            p = learner.predict(x)
        
	if (holdafter and date > holdafter) or (holdout and t % holdout == 0):
            # step 2-1, calculate validation loss
            #           we do not train with the validation data so that our
            #           validation loss is an accurate estimation
            #
            # holdafter: train instances from day 1 to day N
            #            validate with instances from day N + 1 and after
            #
            # holdout: validate with every N instance, train with others
            
            if date != None:
                loss += logloss(p, y)
                count += 1
        elif date != None:
        #elif date >= holdbefore:
            # step 2-2, update learner with label (click) information
            learner.update(x, p, y)
        if t > 10000:
            break

        if t % 2500000 == 0 and t > 1 and count != 0:
            print(' %s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), t, loss/count))
    	
    print('Epoch %d finished, validation logloss: %f, elapsed time: %s' % (
        e, loss/count, str(datetime.now() - start)))


##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
'''
with open(submission, 'w') as outfile:
    outfile.write('id,click\n')
    for t, date, ID, x, y in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (ID, str(p)))
'''
