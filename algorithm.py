# CSC320 Winter 2018
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################

    length = source_patches.shape[1]
    height = source_patches.shape[0]

    source_patches = np.nan_to_num(source_patches)
    target_patches = np.nan_to_num(target_patches)

    k = len(f_heap[0][0])
    f_k = NNF_heap_to_NNF_matrix(f_heap)[0]
    D_k = NNF_heap_to_NNF_matrix(f_heap)[1]

    # Create a matrix for w*alpha*R
    # Use matrix operations instead of looping to improve efficiency
    waR = np.zeros((0, 2))
    # Loop through i's until w*alpha^i is below 1 pixel, record the largest i
    walphai = w
    imax = 0
    while walphai >= 1:
        waR = np.insert(waR, imax, np.ones((1, 2)) * walphai, axis=0)
        imax += 1
        walphai *= alpha
    # Add one line at the end of the matrix with value [0, 0] as for original offsets
    waR = np.insert(waR, imax, np.zeros((1, 2)), axis=0)
    # Note the matrix has imax+1 rows
    n_row = imax + 1
    # Repeat waR k times to do random search on the k patches in heap
    waR = np.repeat(np.array([waR]), k, axis=0).reshape((n_row * k, 2))

    # In odd iterations, offsets are examined in scan order (left to right, top to bottom).
    # Set the loop start = the first pixel, end = the last pixel and step = 1.
    # The top patch and the left patch have known offsets f(y-1,x), f(y,x-1),
    # thus, set alt = -1.
    # In even iterations, set start = the last, end = the first and step = -1.
    # Set alt = 1 to get the known offsets f(y+1,x), f(y,x+1).
    if odd_iteration:
        x_start = 0
        x_end = length - 1
        y_start = 0
        y_end = height - 1
        step = 1
        alt = -1
    else:
        x_start = length - 1
        x_end = 0
        y_start = height - 1
        y_end = 0
        step = -1
        alt = 1

    for x in range(x_start, x_end, step):
        for y in range(y_start, y_end, step):

            if propagation_enabled:
                # The known offsets (eliminating out-of-border indices)
                offset = np.array([])
                if x != x_start:
                    offset = f_k[:, y, x + alt]
                if y != y_start:
                    if not len(offset):
                        offset = f_k[:, y + alt, x]
                    else:
                        offset = np.concatenate((offset, f_k[:, y + alt, x]), axis=0)
                if len(offset):
                    # Indices of the kx2 known target patches
                    tgt_idx = offset + [y, x]
                    tgt_y = np.clip(tgt_idx[:, 0], 0, height-1).astype(int)
                    tgt_x = np.clip(tgt_idx[:, 1], 0, length-1).astype(int)
                    # Patch distances (sum-of-squared-distance)
                    difference = target_patches[tgt_y, tgt_x] - source_patches[y, x]
                    Dv = np.sum(np.linalg.norm(difference, axis=2), axis=1)
                    # Loop to push elements with lower distance into the heap and pop the largest from the heap
                    for i in range(len(Dv)):
                        j = k
                        while Dv[i] < D_k[j-1, y, x] and j > 0:
                            j -= 1
                        if j < k:
                            displace = (tgt_y[i] - y, tgt_x[i] - x)
                            if displace not in f_coord_dictionary[y][x]:
                                D_k[:, y, x] = np.delete(np.insert(D_k[:, y, x], j, Dv[i], axis=0), -1, axis=0)
                                f_k[:, y, x] = np.delete(np.insert(f_k[:, y, x], j, np.array(displace), axis=0), -1, axis=0)
                                max_elmt = heappushpop(f_heap[y][x], (-Dv[i], _tiebreaker.next(), np.array(displace)))
                                f_coord_dictionary[y][x].pop(tuple(max_elmt[2]), None)
                                f_coord_dictionary[y][x][displace] = 1

            if random_enabled:
                # The k offsets in the heap. Each repeat n_row times.
                v0 = np.repeat(f_k[:, y, x], n_row, axis=0)
                # Define ui's
                R = np.random.uniform(-1, 1, (n_row * k, 2))
                u = waR * R + v0
                # Target indices
                tgt_idx = u + np.array([y, x])
                tgt_y = np.clip(tgt_idx[:, 0], 0, height-1).astype(int)
                tgt_x = np.clip(tgt_idx[:, 1], 0, length-1).astype(int)
                # Patch distances (sum-of-squared-distance)
                difference = target_patches[tgt_y, tgt_x] - source_patches[y, x]
                Du = np.sum(np.linalg.norm(difference, axis=2), axis=1)
                # Loop to push elements with lower distance into the heap and pop the largest from the heap
                for i in range(len(Du)):
                    j = k
                    while Du[i] < D_k[j - 1, y, x] and j > 0:
                        j -= 1
                    if j < k:
                        displace = (tgt_y[i] - y, tgt_x[i] - x)
                        if displace not in f_coord_dictionary[y][x]:
                            D_k[:, y, x] = np.delete(np.insert(D_k[:, y, x], j, Du[i], axis=0), -1, axis=0)
                            f_k[:, y, x] = np.delete(np.insert(f_k[:, y, x], j, np.array(displace), axis=0), -1, axis=0)
                            max_elmt = heappushpop(f_heap[y][x], (-Du[i], _tiebreaker.next(), np.array(displace)))
                            f_coord_dictionary[y][x].pop(tuple(max_elmt[2]), None)
                            f_coord_dictionary[y][x][displace] = 1

    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    source_patches = np.nan_to_num(source_patches)
    target_patches = np.nan_to_num(target_patches)
    k = f_k.shape[0]
    length = source_patches.shape[1]
    height = source_patches.shape[0]

    # The identical mapping g(y,x) = [y,x]
    id_map = make_coordinates_matrix((height, length))
    # The source array with dimension kxNxMx2
    src_idx = np.repeat(np.array([id_map]), k, axis=0)
    # The target array and indices
    tgt_idx = src_idx + f_k
    tgt_y = np.clip(tgt_idx[..., 0], 0, height-1)
    tgt_x = np.clip(tgt_idx[..., 1], 0, length-1)
    # Patch distances (sum-of-squared-distance)
    difference = target_patches[tgt_y, tgt_x] \
                 - source_patches[src_idx[..., 0], src_idx[..., 1]]
    Dv_k = np.sum(np.linalg.norm(difference, axis=-1), axis=-1)

    # Loop through the 2D array and create the heap and the dictionary for each pixel
    f_heap = list()
    f_coord_dictionary = list()
    for y in range(height):
        heap_row = list()
        dict_row = list()
        for x in range(length):
            heap = []
            dict = {}
            for i in range(k):
                # Use the negative distance as the priority so as to get a 'MAX' heap
                heappush(heap, (-Dv_k[i, y, x], _tiebreaker.next(), f_k[i, y, x]))
                # Use a tuple as the key of the dictionary
                dict[tuple(f_k[i, y, x])] = 1
            heap_row.append(heap)
            dict_row.append(dict)
        f_heap.append(heap_row)
        f_coord_dictionary.append(dict_row)

    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    heap = copy(f_heap)

    k = len(heap[0][0])
    height = len(heap)
    length = len(heap[0])

    f_k = np.zeros((k, height, length, 2))
    D_k = np.zeros((k, height, length))

    for y in range(height):
        for x in range(length):
            tuples = nlargest(k, heap[y][x])
            for i in range(len(tuples)):
                # Change the distance back to positive value
                D_k[i, y, x] = -tuples[i][0]
                f_k[i, y, x] = tuples[i][2]

    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    length = target.shape[1]
    height = target.shape[0]
    denoised = np.zeros((height, length, 3))

    for x in range(length):
        for y in range(height):
            # The negative distance stored in the first element of each heap entry
            neg_dist = np.array([])
            i = 0
            for tuple in f_heap[y][x]:
                if i == 0:
                    tgt_idx = np.array([tuple[2] + [y, x]])
                else:
                    tgt_idx = np.insert(tgt_idx, i, tuple[2] + [y, x], axis=0)
                neg_dist = np.insert(neg_dist, i, tuple[0], axis=0).astype(float)
                i += 1
            tgt_y = np.clip(tgt_idx[:, 0], 0, height - 1).astype(int)
            tgt_x = np.clip(tgt_idx[:, 1], 0, length - 1).astype(int)
            e = np.exp(neg_dist / h**2)
            Z = np.sum(e)
            # Weight of each pixel depending on how similar they are. Dimension: kx1
            w = e / Z
            # Extend weight to dimension kx3 corresponding to 3 color channels
            w = np.dstack((w, w, w))[0]
            # The color values at each pixel in the heap. Dimension: kx3 -- k pixels, 3 color channels
            col = target[tgt_y, tgt_x]
            # Assign the new color value to the pixel (y,x)
            denoised[y, x] = np.sum(col * w, axis=0)

    #############################################

    return denoised




#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################


#############################################



# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################

    height = target.shape[0]
    length = target.shape[1]
    # The source indices which is an identical mapping g(y,x) = [y,x]
    src_idx = make_coordinates_matrix((height, length))
    # The target indices
    tgt_idx = src_idx + f
    tgt_y = np.clip(tgt_idx[..., 0], 0, height-1).astype(int)
    tgt_x = np.clip(tgt_idx[..., 1], 0, length-1).astype(int)
    # Reconstruct the source patches
    rec_source = target[tgt_y, tgt_x]

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
