import numpy as np
from skimage.io import imread, imsave
from numpy import linalg as LA
from scipy.spatial.transform import Rotation

import scipy
import math #for atan
import numpy as np
from scipy.ndimage import gaussian_filter

#doge1--Doge's Palace Light--I01
A1 = np.float32([[-0.005896293,  0.003201143,  0.000909221],
                 [0.003201143,  0.007083686,  -0.002286545],
                 [0.000909221,  -0.002286545,  0.034119957]])
b1 = np.float32([-0.227557077,  -0.047123300,  0.205254578]).T
c1 = 0.673823976

A2= np.float32([[-0.006897805,  0.002766256,  0.003926461],
                [0.002766256,  0.008716128,  -0.001902301],
                [0.003926461,  -0.001902301,  0.032048057]])    
b2 = np.float32([-0.234017921,  -0.046864423,  0.214483105]).T
c2 = 0.650575084

A3 = np.float32([[-0.007922342,  0.002600586,  0.007483184],
                 [0.002600586,  0.010639797,  -0.001596472],
                 [0.007483184,  -0.001596472,  0.029887898]])
                 
b3 = np.float32([-0.246965886,  -0.047780764,  0.229818354]).T
c3 = 0.642377030

e = 0.001
n_0 = np.array([0,0,1], dtype = float)
max_iter = 50


#return r such that n_0 = R x n
def calc_R(n):

    # uvw = np.cross(n, n_0)
    # rcos = np.dot(n, n_0)
    # rsin = LA.norm(uvw)
    # if not np.isclose(rsin, 0):
        # uvw /= rsin
    # u, v, w = uvw
    
    # tmp = np.array([[ 0, -w,  v],
                    # [ w,  0, -u],
                    # [-v,  u,  0]])
    # r =  rcos * np.eye(3) + rsin * tmp + (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    
    
    
    #R @ n = n_0
    r = Rotation.align_vectors(n_0.reshape(1,3), n.reshape(1,3) )
    R = r[0].as_matrix()

    return R



def shade_x(n):
    s = np.zeros((3,1), dtype = float)
    s[0,:] = n @ A1 @ n.T + b1.T @ n.T + c1
    s[1,:] = n @ A2 @ n.T + b2.T @ n.T + c2
    s[2,:] = n @ A3 @ n.T + b3.T @ n.T + c3
    return s
    
    
def calc_f(n, I):
    #apply shading function 
    s = shade_x(n).copy()
    f = np.zeros((3,1), dtype = float)
    
    f = s - I
    return f
    
def jacob_n(n):
    Jn = np.zeros((3,3), dtype=float)
    Jn[0, :] = n @ A1 + b1.T 
    Jn[1, :] = n @ A2 + b2.T
    Jn[2, :] = n @ A3 + b3.T
    return Jn
    
def jacob_uv(n, n_uv, R):
    Jn = jacob_n(n)
    u, v, r = n_uv
    nuv = np.array([[1, 0],
                    [0, 1],
                    [-u/r, -v/r]])
    Juv = Jn @ R @ nuv
    return Juv
    
def dogleg(f, jac, radius):
    gn_d = -LA.pinv(jac.T @ jac) @ jac.T @ f
    sd_d = -jac.T @ f
    t = -(sd_d.T@jac.T@f)/np.sum((jac@sd_d)**2)
    if LA.norm(gn_d)<= radius:
        return gn_d
    elif t*LA.norm(sd_d)>radius:
        return radius*sd_d/LA.norm(sd_d)
    else:
        pd = t*sd_d
        pf = gn_d-pd
        a = np.sum(pf**2)
        b = np.sum(pd*pf*2)
        c = np.sum(pd**2)-radius
        s = (-b+np.sqrt(b**2-4*a*c))/(2*a)
        return pd+s*pf  
    
def G_N(I_x):
    #init guess R @ n_uv = n maps to n_0
    n_uv = n_0.copy()

     
    #other init guess
    # n1 = np.sqrt(0.3)
    # n2 = np.sqrt(0.3)
    # n = np.array([n1,n2, np.sqrt(1 - (n1**2 + n2**2))])
    # R = LA.inv(calc_R(n))

    n1 = np.random.random_sample() - 0.5
    n2 = np.random.random_sample() - 0.5
    n = np.array([n1, n2, np.sqrt(1-n1**2-n2**2)])
    R = LA.inv(calc_R(n))


    # n = R @ n_uv
    # print(n)
    # R = LA.inv(calc_R(n))

    err = 0
    iter = 0

    f = calc_f(n, I_x)  
    err = LA.norm(f)

    # while(err > e and iter < max_iter):
    #     iter = iter + 1
    #     Juv = jacob_uv(n, n_uv, R)
    #     #update
    #     # h = -LA.pinv(Juv.T @ Juv) @ Juv.T @ f
    #     h = dogleg(f, Juv, 0.1)
    #     uv = n_uv[0:2] + h.T
    #     u, v = uv[0,0], uv[0,1]
    #     uv2 = (u**2 + v**2)
    #     r = np.sqrt(1 - uv2)
    #     n_uv[0] = u
    #     n_uv[1] = v
    #     n_uv[2] = r
    #     if uv2 >= 0.5:
    #         n = R @ n_uv
    #         R = LA.inv(calc_R(n))
    #         n_uv = n_0.copy()
    #     else:
    #         #print("r:" , r)
    #         n = (R @ n_uv.T).T
    #         #print(n)   
    #     f = calc_f(n, I_x)
    #     err = LA.norm(f)
    #     #print("E:", err)
    # #print(iter)
    while(err > e and iter < max_iter):
        iter = iter + 1 
        Juv = jacob_uv(n, n_uv, R)
        #update
        #h = -1.0 * LA.pinv(Juv.T @ Juv) @ Juv.T @ f
        h = dogleg(f, Juv, 0.1)
        #print("h:", h)
        n_uv[0:2] = n_uv[0:2] + h.T
        u, v, r = n_uv
        #print("hu:", u)
        #print("hv:", v)
        #reset
        if (u**2 + v**2) >= 0.5:
            #new guess
            n = R @ n_uv
            #n = n/LA.norm(n)
            R = LA.inv(calc_R(n))
            #print(R)
            #n_uv
            n_uv = n_0.copy()
            
        else: 
            r = np.sqrt(1 - (u**2 + v**2))
            #print("r:" , r)
            n_uv[2] = r
            n = (R @ n_uv.T).T  
            #print(n)

        f = calc_f(n, I_x)
        err = LA.norm(f)
        #print("E:", err) 
    return n

#iterate through each pixel         
def natural_SFS(img, mask):
    h, w, c = img.shape 
    print(img.shape)
    nrm = np.zeros(img.shape)
    for i in range(h):
        print(i)
        for j in range(w):
            if (mask[i, j] == 1):
                nrm[i, j, :] = G_N(img[i, j, :].reshape(3,1))
    return nrm
    
    
    
########################## Support code below

# from os.path import normpath as fn # Fixes window/linux path conventions
# import warnings
# warnings.filterwarnings('ignore')


# # Utility functions to clip intensities b/w 0 and 1
# # Otherwise imsave complains
# def clip(im):
    # return np.maximum(0.,np.minimum(1.,im))


# ############# Main Program -np.sqrt(0.5)








from os.path import normpath as fn # Fixes window/linux path conventions
import warnings

# #simple test of G_N of a single point
# n_test = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0  ], dtype = float)
# I_test = shade_x(n_test)
# print("n_test", n_test)
# print("I_test", I_test.T)

# n_r = G_N(I_test).T
# print("result", n_r)
# print("shade result", shade_x(n_r).T)

#Load image data
imgs = []

for i in range(1,11):
    imgs = imgs + [np.float32(imread(fn('test/inputs/im_b04_%02d.png' % i)))/255.]
    
    
#print(imgs[1].shape)
img = imgs[0]
mask = np.float32(imread(fn('test/blob04_mask.png')) > 0)

nrm = (np.float32(imread(fn('test/blob04_normal.png')))/255. - 0.5) * 2

# n_test = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0  ], dtype = float)
# I_test = shade_x(n_test)
# print("n_test", n_test)
# print("I_test", I_test.T)

# print("I_test.shape", I_test.shape)

# n_r = G_N(I_test).T
# print("result", n_r)
# print("shade result", shade_x(n_r).T)


#patch data
I_patch = img[93:96, 80:83,]
nrm_patch = nrm[93:96, 80:83,]

I_test = I_patch[0,1,]

n_test = nrm_patch[0,1,]


# n_r = G_N(I_test.reshape(3,1)).T
# I_r = shade_x(n_r)
# print("I_test", I_test.T)
# print("n_test", n_test)
# print("result_n", n_r)
# print("result_I", I_r)
# # n_patch = np.zeros((9,3), dtype=float)

# # I_patch_flatten = I_patch.reshape(9,3)
# # for i in range(9):
#     # n_patch[i,] = G_N(I_patch[i,])
# # print(n_patch)

# n_single = nrm_patch.reshape(9,3)[0,]
# print("n_single\n", n_single)
# I_check_single = shade_x(n_single)
# print("I_check_single\n", I_check_single)




#iterate through each pixel         
def natural_SFS(img, mask):
    h, w, c = img.shape
    print(img.shape)
    nrm = np.zeros(img.shape)
    for i in range(h):
        #print(i)
        for j in range(w):
            if (mask[i, j] == 1):
                #print(img[i, j, :].shape)
                nrm[i, j, :] = G_N(img[i, j, :].reshape(3,1)).T
    return nrm
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

print("nrm test", nrm[93:96, 80:83,])
nrm = natural_SFS(img, mask)
print("nrm", nrm[93:96, 80:83,])
nimg = (nrm/2.0+0.5)
print("nimg", nimg[93:96, 80:83,])
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('test/outputs/my_nrm.png'),nimg)






