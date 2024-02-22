#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    int w = (int)ceil(sigma*6);
	if (w % 2 == 0) {
		w++;
	}
    image filter = make_image(w,1,1);
	float* g = filter.data;
    for (int x = 0; x < w; x++) {
        float mut_x = x - w/2;
        *g = exp(-(pow(mut_x, 2))/(2*powf(sigma, 2)))/(sqrtf(TWOPI)*sigma);
        g++;
    }

	l1_normalize(filter);
    return filter;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    image gaus = make_1d_gaussian(sigma);
    image gaus2 = make_image(1,gaus.w,1);
    memcpy(gaus2.data, gaus.data, gaus.w*sizeof(float));

    image copy = convolve_image(im, gaus, 1);
    copy = convolve_image(copy, gaus2, 1);
    return copy;
}

// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
    image s = make_image(im.w, im.h, 3);

    // derivatives
    float* gx = convolve_image(im, make_gx_filter(), 0).data;
    float* gy = convolve_image(im, make_gy_filter(), 0).data;

    // measures
    for (int i = 0; i < im.h*im.w; i++) {
        s.data[i] = (gx[i]*gx[i]);
        s.data[i+im.h*im.w] = (gy[i]*gy[i]);
        s.data[i+2*im.h*im.w] = (gx[i]*gy[i]);
	}

    // weight
    image fin = smooth_image(s, sigma);
    return fin;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    image R = make_image(S.w, S.h, 1);
    for (int i = 0; i < S.h*S.w; i++) {
        float d = S.data[i] * S.data[i+S.h*S.w] - S.data[i+2*S.h*S.w]*S.data[i+2*S.h*S.w];
        float t = S.data[i] + S.data[i+S.h*S.w];
        R.data[i] = (d) - .06 * pow(t, 2);
	}
    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{
    image r = copy_image(im);
    for (int h = 0; h < im.h; h++) {
        for (int i = 0; i < im.w; i++) {
            float v = get_pixel(im, i, h, 0);

            // neighbor range
            for (int j = -w; j < w+1; j++) {

                // height
                int y = fmin(im.h - 1, h+j);
                y = fmax(0, y);

                for (int k = -w; k < w+1; k++) {

                    // width
                    int x = fmin(im.w - 1, i+k);
                    x = fmax(0, x);

                    float v2 = get_pixel(im, x, y, 0);
                    if (v2 > v) {
                        set_pixel(r, i, h, 0,-99999);
                    }
                }
            }
        }
	}
    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);


    //TODO: count number of responses over threshold
    int count = 0;
    for (int i = 0; i < Rnms.w*Rnms.h*Rnms.c; i++) {
        if (thresh < Rnms.data[i]) {
            count++;
        }
    }


    *n = count; // <- set *n equal to number of corners in image.
    descriptor *d = calloc(count, sizeof(descriptor));
    int x = 0;

    for (int i = 0; i < im.w*im.h; i++) {
        if (thresh < Rnms.data[i]) {
            d[x++] = describe_index(im, i);
        }
    }

    free_image(S);
    free_image(R);
    free_image(Rnms);
    return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
