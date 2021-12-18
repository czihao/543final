import scipy
import math #for atan
import numpy as np
from scipy.ndimage import gaussian_filter
from numpy import linalg as LA
from scipy.spatial.transform import Rotation
from skimage.io import imread, imsave

#####################################################################
#Light conditions
#####################################################################
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
                 
b3 = np.float32([-0.246965886,  -0.047780764,  0.229818354])
c3 = 0.642377030


#put light into lists
L_A = [np.float32([[-0.005896293,  0.003201143,  0.000909221],
                 [0.003201143,  0.007083686,  -0.002286545],
                 [0.000909221,  -0.002286545,  0.034119957]]),
                 
     np.float32([[-0.006897805,  0.002766256,  0.003926461],
                [0.002766256,  0.008716128,  -0.001902301],
                [0.003926461,  -0.001902301,  0.032048057]]),

     np.float32([[-0.007922342,  0.002600586,  0.007483184],
                 [0.002600586,  0.010639797,  -0.001596472],
                 [0.007483184,  -0.001596472,  0.029887898]])]
                 
L_b = [np.float32([-0.227557077,  -0.047123300,  0.205254578]),
       np.float32([-0.234017921,  -0.046864423,  0.214483105]),
       np.float32([-0.246965886,  -0.047780764,  0.229818354])
       ]
       
L_c = [0.673823976, 0.650575084, 0.642377030]
k = 9
#0.01~0.5
lamda1 = 0.01
lamda2 = 0
e = 0.001
max_iter = 50

n_map = np.array([0,0,1], dtype = float)
n_map_patch = np.zeros((9,3), dtype=float)
n_map_patch[:,] = n_map

uvr_0 = np.array([0,0,1], dtype = float)

###############################################################
#Second derivative gaussian of the whole image
###############################################################

#have not integrated results from previous scale
#this can be done to the whole image, img's dim = LxW
#return G2 for the whole image of dim = LXWX3X3,
def compute_G2(img):
    L, W, Z= img.shape 
    img_grey = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3
    #vertical Iy
    I_y = np.gradient(img_grey, axis = 0)
    sqr_I_y = I_y**2
    
    #horizontal Ix
    I_x = np.gradient(img_grey, axis = 1)
    sqr_I_x = I_x**2
    
    #sigma = 1
    G_x = gaussian_filter(I_x, 1)
    
    G_y = gaussian_filter(I_y, 1)
    
    Gxy = gaussian_filter(I_x*I_y, 1)
    
    C = (G_x - G_y)/(G_x + G_y + 0.000001)
    
    S = (2 * Gxy)*(G_x + G_y)
    
    #local orientation in radians
    theta = 0.5 * np.arctan2(S, C)
    #print("theta.shape", theta.shape)
    ka = np.cos(theta)**2
    #print("ka.shape", ka.shape)
    kb = -2 * np.cos(theta) * np.sin(theta)
    #print("kb.shape", kb.shape)
    kc = np.sin(theta)**2
    #print("kc.shape", kc.shape)
    
    x = np.arange(-1, 2, 1)
    y = np.arange(-1, 2, 1)
    xx, yy = np.meshgrid(x, y)
    G_2a = 0.9213 * (2 * x ** 2 - 1) * np.exp(-(xx**2 + yy**2))
    #print("G_2a.shape", G_2a.shape)
    G_2b = 1.843 * xx * yy * np.exp(-(xx**2 + yy**2))
    #print("G_2b.shape",G_2b.shape)
    G_2c = 0.9213 * (2 * y ** 2 - 1) * np.exp(-(xx**2 + yy**2))
    #print("G_2c.shape",G_2c.shape)
    
    #select the theta at the middle of the patch, G2's dib  m = LxWx3x3
    #t = np.einsum('i,jk->ijk', ka.reshape(W*L), G_2a)
    G2 = (np.einsum('i,jk->ijk', ka.reshape(W*L), G_2a) + np.einsum('i,jk->ijk', kb.reshape(W*L) ,G_2b) \
            + np.einsum('i,jk->ijk', kc.reshape(W*L), G_2c)).reshape(L,W,3,3)
     
    return G2


##################################################################
#Rotation matrix calculation
##################################################################

#return r such that n_0 = R x n
def calc_R(n):
    uvw = np.cross(n, n_map)
    rcos = np.dot(n, n_map)
    rsin = LA.norm(uvw)
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw
    
    tmp = np.array([[ 0, -w,  v],
                    [ w,  0, -u],
                    [-v,  u,  0]])
    r =  rcos * np.eye(3) + rsin * tmp + (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    
    return r
    
def calc_R_Patch(n):
    #n_flatten = n.reshape(9,1,3) for rotation
    n_flatten = n.reshape(9,3)
    R = np.zeros((9,3,3), dtype=float)
    
    for i in range(9):
        r = calc_R(n_flatten[i,:])
        #r = Rotation.align_vectors(n_map.reshape(1,3), n_flatten[i,:])
        #R[i,:] = r[0].as_matrix()
        R[i,:] = r
    return R

   
##################################################################
#shading and f_cost
##################################################################
def shade_patch(uvr_patch, R):
    #npatch flatten to 9x3
    uvr_patch_flatten = uvr_patch.reshape(9,3)
    R_flatten = R.reshape(9,3,3)

    #get R@N, the true norm
    npatch_flatten = np.matmul(R_flatten.reshape(9,3,3),uvr_patch_flatten.reshape(9,3,1)).swapaxes(1,2)

    S = np.zeros((3 * 9,1), dtype = float)
    
        
    for i in range(9):
        S[i*3,:] = npatch_flatten[i,:] @ A1 @ npatch_flatten[i,:].T + b1.T @ npatch_flatten[i,:].T + c1
        S[i*3 + 1,:] = npatch_flatten[i,:] @ A2 @ npatch_flatten[i,:].T + b2.T @ npatch_flatten[i,:].T + c2
        S[i*3 + 2,:] = npatch_flatten[i,:] @ A3 @ npatch_flatten[i,:].T + b3.T @ npatch_flatten[i,:].T + c3
    
    #S is suppoesd to be 27x1
    return S
    
    
#wrong dimension now, ingore it
def f_cost(S, I):
    I_X = I.reshape(27,1)
    f_cost = S - I_X
    return f_cost


###############################################################
# cost of C1 & C2
###############################################################

#return vector with the curl for each n1 to nk in seperate rows
#npatch is the uv of nrms of the patch, R is the rotation matrix of the patch
#C1 should be kx1, with each row the value of curl of the norm
def integrability_cost(patch_nrm, R):
    uv = patch_nrm.reshape(3,3,3)[:,:, 0:2]
    
    npatch_flatten = patch_nrm.reshape(k,3)
    R_flatten = R.reshape(k,3,3)
    rs_y = R_flatten[:, 0, 0:2].reshape(k,2)
    rs_x = R_flatten[:, 1, 0:2].reshape(k,2)
    
    #can be sub to patch_w and patch_h
    #TODO: boundary correct?
    uv_y = np.zeros((3,3,2), dtype=float)
    uv_y[0:2,:,:] = uv[1:3,:,:] - uv[0:2,:,:]
    uv_y[2,:,:] = - uv[2,:,:]
    uv_y_flatten = uv_y.reshape(k,2)
    
    
    uv_x = np.zeros((3,3,2), dtype=float)
    uv_x[:,0:2,:] = uv[:,1:3,:] - uv[:,0:2,:]
    uv_x[:,2,:] = -uv[:,2,:]
    uv_x_flatten = uv_x.reshape(k,2)
    
    c_y = np.sum(uv_y_flatten * rs_y, axis = 1)
    c_x = np.sum(uv_x_flatten * rs_x, axis = 1)
    
    curl = np.zeros((k,1), dtype = float)
    
    curl = c_y - c_x
    
    curl = lamda1 * curl
    return curl

#compute only for a patch, the patch_G2 is the 3x3 G2 of the pixel in the middle of the patch
#C2 should be 2x1
def smoothness_cost(patch_G2, patch_nrm):
    u = patch_nrm.reshape(3,3,-1)[:,:, 0] 
    v = patch_nrm.reshape(3,3,-1)[:,:, 1]
    
    C2 = np.zeros((2,1), dtype=float)
    
    #might need sum twice for both axes
    C2[0,0] = np.sum(patch_G2 * u)
    C2[1,0] = np.sum(patch_G2 * v)
    C2 = lamda2 * C2
    return C2


##############################################################
#compute g cost
##############################################################

#n_patch.dim = 9*3, I_patch.dim = 9*3, R_patch = 9*3*3
def calc_g(uvr_patch, I_patch, R_patch, G2_patch = 0):  #, G2_patch
    g = np.zeros((38,1), dtype=float)
    
    S = shade_patch(uvr_patch, R_patch)
    #f.dim = 27*1, c1.dim = 9*1, c2.dim = 2*1
    f = f_cost(S, I_patch)
    g[0:27,] = f
    
    c1 = integrability_cost(uvr_patch, R_patch).reshape(9,1)
    #c2 = smoothness_cost(G2_patch, uvr_patch)
    #g.dim = 38*1
    #g = np.concatenate((f, c1), axis=0)
    g[27:36,] = c1
    
    return g


##############################################################
#Jacobian Computation for f
##############################################################

#input is n = R(u, v, r)
#input n_patch.dim 9x1x3
def jacob_n_patch(n_patch):
    n_patch_flatten = n_patch.reshape(9,3)
    Jn = np.zeros((27,3), dtype=float)
    for i in range (3):
        Jn[i:27:3, :] = n_patch_flatten @ L_A[i] + L_b[i].T 
    return Jn
    
    
#can be changed to whole img
#n_patch.dim 9x1x3, n_uv_patch.dim = 9x1x3, R_patch.dim = 9x3x3
def jacob_uv_patch(n_patch, n_uv_patch, R_patch):
    Jn = jacob_n_patch(n_patch) #Jn.dim = 9x3x3
    Jn = Jn.reshape(9,3,3)
    #print(Jn.shape)
    n_uv_patch_flatten = n_uv_patch.reshape(9,3)
    
    nuv = np.zeros((9,3,2),  dtype=float)
    nuv[:, 0:2 , :] = np.array([[1,0],[0,1]])
    
    nuv[:, 2 , 0] =  (n_uv_patch_flatten[:,0]/ n_uv_patch_flatten[:,2]).reshape(9)
    nuv[:, 2 , 1] =  (n_uv_patch_flatten[:,1]/ n_uv_patch_flatten[:,2]).reshape(9)

    R_flatten = R_patch.reshape(9,3,3)
 
    #Juv.dim = 9x3x2
    Juv = np.matmul(np.matmul(Jn,R_flatten), nuv)
    return Juv.reshape(27,2)

##########################################################
#Jacobian for C1 & C2
##########################################################

#can be changed to whole img
#R_patch = (9x3x3) or (WxLx3x3)
def jacob_c1(R_patch):
    #R = R_patch.reshape(9,3,3)
    R = R_patch.copy()
    #for each location, return (r11 + r21 at u(i,j), r12 + r22 at v(i,j), 
                                #r11 at u(i + 1, j), r12 at v(i + 1, j), -r21 at u(i, j + 1), -r22 at v(i, j + 1)) 
    J_c1 = np.zeros((9,1,18), dtype=float)
    #flatten out
    J_c1 = J_c1.reshape(1,-1)
    #-r11+r21
    J_c1[0, 0::20] = -R[:,0,0] + R[:,1,0]
    #-r12+r22
    J_c1[0, 1::20] = -R[:,0,1] + R[:,1,1]
    #r11
    J_c1[0, 6:108:20] = R[6,0,0]
    #r12
    J_c1[0, 7:108:20] = R[6,0,1]
    #r21
    J_c1[0, 3:8*18:20] = -R[8,1,0]
    #r22
    J_c1[0, 4:8*18:20] = -R[8,1,1]

    #zero out
    J_c1[0, 42:6*18:60] = 0.0 
    J_c1[0, 43:6*18:60] = 0.0 
    
    #dim = 9x18
    J_c1 = J_c1.reshape(9,18)
    J_c1 = lamda1 * J_c1
    return J_c1


#can be changed to whole image?
#patch_G2.dim = 3x3
def jacob_c2(patch_G2):
    
    J_c2 = np.zeros((2,18), dtype=float)
    J_c2[0, 0:18:2] = patch_G2.reshape(1,9)
    J_c2[1, 1:18:2] = patch_G2.reshape(1,9)
    
    J_c2 = lamda2 * J_c2
    #dim = 2 x 18
    return J_c2

##########################################################
#Compute the full Jacobian
##########################################################

#merge all jacobian, and return 38x18
def Jacob(n_patch, n_uv_patch, R_patch, patch_G2 = 0):
    full_Jacob = np.zeros((38,18), dtype=float)

    #call sub jacob
    J_uv = jacob_uv_patch(n_patch, n_uv_patch, R_patch)
    
    J_uv = J_uv.reshape(9,3,2)
    #Juv_sparse.dim 27*18
    Juv_sparse = np.zeros((9, 3, 9, 2), dtype=float)
    row = Juv_sparse.shape[0]
    for i in range(row):
        Juv_sparse[i, :, i, :] = J_uv[i]
    Juv_sparse = Juv_sparse.reshape(27, 18)
    
    J_c1 = jacob_c1(R_patch)
    #J_c2 = jacob_c2(patch_G2)
    full_Jacob[0:27,] = Juv_sparse
    full_Jacob[27:36, ] = J_c1


    return full_Jacob
    
def dogleg(f, jac, radius):
    gn_d = -LA.pinv(jac.T @ jac) @ jac.T @ f
    sd_d = -jac.T @ f
    t = - (sd_d.T@jac.T@f)/np.sum((jac@sd_d)**2)
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

def G_N_patch(I_patch, initial_guess, patch_G2 = 0):
    err = 0
    iter = 0

    #other init guess
    # n1 = np.sqrt(0.3)
    # n2 = np.sqrt(0.3)
    # n1 = np.random.random_sample() - 0.5
    # n2 = np.random.random_sample() - 0.5
    # n_0 = np.array([n1,n2, np.sqrt(1 - (n1**2 + n2**2))])

    # 9 random initial guess
    # n_0_patch = np.zeros((3,3,3), dtype=float)
    # n_0_patch[:,:,0:2] = np.random.random_sample((3,3,2)) - 0.5
    # n_0_patch[:,:,2] = np.sqrt(np.ones((3,3), dtype=float) - n_0_patch[:,:,0]**2 - n_0_patch[:,:,1]**2)
    n_0_patch = initial_guess.copy()
    R_0_patch = LA.inv(calc_R_Patch(n_0_patch))
    n_patch = n_0_patch.reshape(9,1,3)
    R_patch = R_0_patch.reshape(9,3,3)
    uvr_patch = n_map_patch.copy()
    
    #check 
    assert (n_patch == np.matmul(R_patch.reshape(9,3,3),uvr_patch.reshape(9,3,1)).swapaxes(1,2)).all
    
    g = calc_g(uvr_patch, I_patch, R_patch, patch_G2)
    #print("g1\n", g)
    err = LA.norm(g)
    #print("err1\n", err)
    
    while(err > e and iter < max_iter):
        iter = iter + 1
        # print("iter: ", iter)
        full_jacob = Jacob(n_patch, uvr_patch, R_patch, patch_G2)
        #print("jacob\n", full_jacob)
        
        # # update, h.dim = 18x1
        #h = -1.0 * LA.pinv(full_jacob.T @ full_jacob) @ full_jacob.T @ g
        h = dogleg(g, full_jacob, 0.1)
        #print("h\n", h.reshape(9,2))
        
        
        uvr_patch[:,0:2] = uvr_patch[:,0:2]+ h.reshape(9,2)
       
        uvr_update = uvr_patch[:,0]**2 + uvr_patch[:,1]**2
        uvr_bool = uvr_update >= 0.5
        index_reset = np.where(uvr_bool == True)
        #print("index_reset:", index_reset)
        index_update = np.where(uvr_bool == False)
        #print("index_update:", index_update)
        
        #reset
        reset_size = R_patch[index_reset,:,:].shape[1]
        if (reset_size > 0) :
            R_reset = R_patch[index_reset,:,:].reshape(reset_size, 3, 3)
            uvr_reset = uvr_patch[index_reset,:].reshape(reset_size, 3, 1)
            n_newguess = np.matmul(R_reset, uvr_reset).swapaxes(1,2)
            R_update = np.zeros((reset_size,3,3), dtype=float)
            for i in range(reset_size):
                r = Rotation.align_vectors(n_map.reshape(1,3), n_newguess[i,:])
                R_update[i,:] = r[0].as_matrix()
                
            
            R_patch[index_reset,] =  LA.inv(R_update)
            uvr_patch[index_reset,:] = n_map
            n_patch[index_reset,:] = n_newguess
            
        uvr_r = np.sqrt(1 - uvr_update[index_update])
        
        uvr_patch[index_update,2] = uvr_r
        
        n_patch = np.matmul(R_patch.reshape(9,3,3),uvr_patch.reshape(9,1,3).swapaxes(1,2)).swapaxes(2,1)
        #print("n_patch\n", n_patch)
        #print("R_patch\n", R_patch)
        #print("uvr_patch\n", uvr_patch)
        #print("uvr_patch after update\n", uvr_patch)
        
        g = calc_g(uvr_patch, I_patch, R_patch, patch_G2)
        err = LA.norm(g)  
        #print("err\n", err)
    return n_patch



#test code

#get a random R by creating a random n, and the R-1 will map the random n to [0,0,1] 
# n1 = np.sqrt(0.3)
# n2 = np.sqrt(0.3)
# random_n = np.array([n1,n2, np.sqrt(1 - (n1**2 + n2**2))])
# #inv(random_R) @ random_n = [0,0,1]
# random_R = LA.inv(calc_R(random_n))
# n_0 = random_R @ n_map
# R_0 = LA.inv(calc_R(n_0))
# assert (random_n == n_0).all    
# assert (random_R == R_0).all        



from os.path import normpath as fn # Fixes window/linux path conventions
import warnings

#Load image data
imgs = []

for i in range(1,11):
    imgs = imgs + [np.float32(imread(fn('test/inputs/im_b04_%02d.png' % i)))/255.]
    
    
#print(imgs[1].shape)
img = imgs[0]
mask = np.float32(imread(fn('test/blob04_mask.png')) > 0)

nrm = (np.float32(imread(fn('test/blob04_normal.png')))/255. - 0.5) * 2
#nrm = np.float32(imread(fn('test/blob04_normal.png')))/255.
#nrm[:,:,2] = -nrm[:,:,2]


I_patch = img[93:96, 80:83,]
nrm_patch = nrm[93:96, 80:83,]

# print("I_patch size", I_patch.shape)
# print("nrm_patch size", nrm_patch.shape)
# n_patch = G_N_patch(I_patch)
# print("n_test", nrm_patch)
# print("n_result", n_patch.reshape(3,3,3))
# print("image dim", I_patch.shape)

def natural_SFS(img, mask):
    h, w, c = img.shape
    print(img.shape)
    nrm = np.zeros(img.shape)
    #first guess
    next_guess = np.zeros((3, 3, 3), dtype=float)
    new_guess = np.zeros((3, 3, 3), dtype=float)
    new_guess[:,:,0:2] = np.random.random_sample((3,3,2)) - 0.5
    new_guess[:,:,2] = np.sqrt(np.ones((3,3), dtype=float) - new_guess[:,:,0]**2 - new_guess[:,:,1]**2)
    for i in range(h):
        #print(i)
        for j in range(w):
            if (mask[i, j] == 1):
                I_patch_ij = np.zeros((3, 3, 3), dtype=float)
                random_guess = np.zeros((3, 1, 3), dtype=float)
                random_guess[:,:,0:2] = np.random.random_sample((3,1,2)) - 0.5
                random_guess[:,:,2] = np.sqrt(np.ones((3,1), dtype=float) - random_guess[:,:,0]**2 - random_guess[:,:,1]**2)
                for a in range(3):
                    for b in range(3):
                        if (mask[i+a, j+b] == 1):
                            I_patch_ij[a, b, :] = img[i+a, j+b, :]
                        else:
                            I_patch_ij[a, b, :] = np.zeros((1,3), dtype=float)
                assert new_guess.shape == (3,3,3)
                next_guess = G_N_patch(I_patch_ij.reshape(9,3), new_guess).reshape(3,3,3)
                new_guess = np.concatenate((next_guess[:,1:3,:],random_guess), axis=1) 
                assert nrm[i, j, :].shape == next_guess[0, 0, :].shape
                nrm[i, j, :] = next_guess[0, 0, :]
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


