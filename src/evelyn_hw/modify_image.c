#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

/******************************** Resizing *****************************
  To resize we'll need some interpolation methods and a function to create
  a new image and fill it in with our interpolation methods.
************************************************************************/

float nn_interpolate(image im, float x, float y, int c)
{
    // convert floats to nearest int
    int new_y = (int)(round(y));
    int new_x = (int)(round(x));

    // clamp values
    c = fmin(im.c - 1, c);
    c = fmax(0, c);
    new_y = fmin(im.h - 1, new_y);
    new_y = fmax(0, new_y);
    new_x = fmin(im.w - 1, new_x);
    new_x = fmax(0, new_x);


    float* ptr = im.data;
    ptr += (im.h * im.w * c);  // get to proper channel
    ptr += (im.w * new_y);  // get to proper row
    ptr += new_x;  // get to proper column
    return *ptr;
}

image nn_resize(image im, int w, int h)
{
    image copy = make_image(w, h, im.c);
    float aw = ((float)im.w) / w;
    float ah = ((float)im.h) / h;
    float* cpy_ptr = copy.data;

    for (int i = 0; i < copy.c; i++) { // channels
        for (int j = 0; j < copy.h; j++) { // rows
            for (int k = 0; k < copy.w; k++) { // cols
              float mut_w = aw * (k + .5) - .5;
              float mut_h = ah * (j + .5) - .5;
              *cpy_ptr = nn_interpolate(im, mut_w, mut_h, i);
              cpy_ptr++;
            }
        }
    }

    return copy;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    int cx = ceil(x);
    int fx = floor(x);
    int cy = ceil(y);
    int fy = floor(y);

    float v1 = get_pixel(im, fx, fy, c);
    float v2 = get_pixel(im, cx, fy, c);
    float v3 = get_pixel(im, fx, cy, c);
    float v4 = get_pixel(im, cx, cy, c);

    float d1 = (x - fx);
    float d2 = (cx - x);
    float d3 = (y - fy);
    float d4 = (cy - y);

    // find areas
    float a1 = d2*d4;
    float a2 = d1*d4;
    float a3 = d2*d3;
    float a4 = d1*d3;

    return (a1*v1) + (a2*v2) + (a3*v3) + (a4*v4);
}

image bilinear_resize(image im, int w, int h)
{
  // Algorithm is same as nearest-neighbor interpolation.
    image copy = make_image(w, h, im.c);
    float aw = ((float)im.w) / w;
    float ah = ((float)im.h) / h;
    float* cpy_ptr = copy.data;

    for (int i = 0; i < copy.c; i++) { // channels
        for (int j = 0; j < copy.h; j++) { // rows
            for (int k = 0; k < copy.w; k++) { // cols
              float mut_w = aw * (k + .5) - .5;
              float mut_h = ah * (j + .5) - .5;
              *cpy_ptr = bilinear_interpolate(im, mut_w, mut_h, i);
              cpy_ptr++;
            }
        }
    }
    return copy;
}


/********************** Filtering: Box filter ***************************
  We want to create a box filter. We will only use square box filters.
************************************************************************/

void l1_normalize(image im)
{
    float* ptr = im.data;
    float ratio = 0;
	for (int i = 0; i < im.c*im.w*im.h; i++) {
        ratio += im.data[i];
    }
    for (int i = 0; i < im.c*im.w*im.h; i++) {
        *ptr /= ratio;
        ptr++;
    }
}

image make_box_filter(int w)
{
  image f = make_image(w, w, 1);
  float* f_ptr = f.data;
  for (int i = 0; i < w * w; i++) {
    *f_ptr = 1;
    f_ptr++;
  }
  l1_normalize(f);
  return f;
}

image convolve_image(image im, image filter, int preserve)
{
    assert(im.c == filter.c || filter.c == 1);

	int midfw = (filter.w / 2);
    int midfh = (filter.h / 2);
    image copy;
	if (preserve) {
		copy = make_image(im.w, im.h, im.c);
	} else {
		copy = make_image(im.w, im.h, 1);
	}
    float* cpy_ptr  = copy.data;

    for (int i = 0; i < copy.c; i++) { // channels
        for (int j = 0; j < copy.h; j++) { // rows
            for (int k = 0; k < copy.w; k++) { // cols
				float val = 0.0f;

				// Using the filter
				for (int b = 0; b < filter.h; b++) { // rows
					for (int c = 0; c < filter.w; c++) { // cols
						int x = k + c - midfw;
						int y = j + b - midfh;

						if (preserve) {
							if (filter.c == 1) {
								val += get_pixel(filter, c, b, 0)*get_pixel(im, x, y, i);
							} else {
								val += get_pixel(filter, c, b, i)*get_pixel(im, x, y, i);
							}
						} else {
							// collapse any other layers to 1 layer
							if (filter.c == 1) {
								for (int d = 0; d < im.c; d++) {
									val += get_pixel(filter, c, b, 0)*get_pixel(im, x, y, d);
								}
							} else {
								for (int d = 0; d < im.c; d++) {
									val += get_pixel(filter, c, b, d)*get_pixel(im, x, y, d);
								}
							}
						}
					}
				}
              	*cpy_ptr = val;
    	        cpy_ptr++;
            }
        }
    }
    return copy;
}

image make_highpass_filter()
{
    image f = make_image(3, 3, 1);
    f.data[0] = 0;
    f.data[1] = -1;
    f.data[2] = 0;
    f.data[3] = -1;
    f.data[4] = 4;
    f.data[5] = -1;
    f.data[6] = 0;
    f.data[7] = -1;
    f.data[8] = 0;
    return f;
}

image make_sharpen_filter()
{
    image f = make_image(3, 3, 1);
    f.data[0] = 0;
    f.data[1] = -1;
    f.data[2] = 0;
    f.data[3] = -1;
    f.data[4] = 5;
    f.data[5] = -1;
    f.data[6] = 0;
    f.data[7] = -1;
    f.data[8] = 0;
    return f;
}

image make_emboss_filter()
{
    image f = make_image(3, 3, 1);
    f.data[0] = -2;
    f.data[1] = -1;
    f.data[2] = 0;
    f.data[3] = -1;
    f.data[4] = 1;
    f.data[5] = 1;
    f.data[6] = 0;
    f.data[7] = 1;
    f.data[8] = 2;
    return f;
}

// Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: sharp and emboss. The highpass filter looks fairly monochromatic whether or not you preserve whereas sharpen and emboss
// have color if you preserve the channels (which only become exaggerated with the filters, which looks better overall).
// Using the highpass filter with layer preservation also loses some of the lines, making the image less readable.

// Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: Yes because when we are doing convolution with these filters, we are multiplying
// our values by negatives or numbers greater than 1, so there should be some clamping to get float values
// between 0 and 1.

image make_gaussian_filter(float sigma)
{
	int w = (int)ceil(sigma*6);
	if (w % 2 == 0) {
		w++;
	}
	image filter = make_image(w,w,1);

	float* g = filter.data;
	for (int y = 0; y < w; y++) {
		for (int x = 0; x < w; x++) {
				float mut_x = x - w/2;
            	float mut_y = y - w/2;
			*g = exp(-(pow(mut_x, 2)+pow(mut_y, 2))/(2*powf(sigma, 2)))/(TWOPI*powf(sigma, 2));
			g++;
		}
	}
	l1_normalize(filter);
    return filter;
}

image add_image(image a, image b)
{
	assert(a.h == b.h && a.w == b.w && a.c == b.c);
	image copy = make_image(a.w,a.h,a.c);

	float* a_ptr = a.data;
	float* b_ptr = b.data;
	float* c_ptr = copy.data;

    for (int i = 0; i < a.c*a.h*a.w; i++) {
        c_ptr[i] = (a_ptr[i]) + (b_ptr[i]);
    }
    return copy;
}

image sub_image(image a, image b)
{
	assert(a.h == b.h && a.w == b.w && a.c == b.c);
	image copy = make_image(a.w,a.h,a.c);

	float* a_ptr = a.data;
	float* b_ptr = b.data;
	float* c_ptr = copy.data;

    for (int i = 0; i < a.c*a.h*a.w; i++) {
        c_ptr[i] = (a_ptr[i]) - (b_ptr[i]);
    }
    return copy;
}

image make_gx_filter()
{
    image f = make_image(3, 3, 1);
    f.data[0] = -1;
    f.data[1] = 0;
    f.data[2] = 1;
    f.data[3] = -2;
    f.data[4] = 0;
    f.data[5] = 2;
    f.data[6] = -1;
    f.data[7] = 0;
    f.data[8] = 1;
    return f;
}

image make_gy_filter()
{
    image f = make_image(3, 3, 1);
    f.data[0] = -1;
    f.data[1] = -2;
    f.data[2] = -1;
    f.data[3] = 0;
    f.data[4] = 0;
    f.data[5] = 0;
    f.data[6] = 1;
    f.data[7] = 2;
    f.data[8] = 1;
    return f;
}

void feature_normalize(image im)
{
	float max = im.data[0];
	float min = im.data[0];

	for (int i = 1; i < im.c*im.h*im.w; i++) {
        if(im.data[i] > max) max = im.data[i];
        if(im.data[i] < min) min = im.data[i];
    }
	float diff = max - min;

	if (diff == 0) {
		for (int i = 0; i < im.c*im.h*im.w; i++) {
			im.data[i] = 0;
		}
	} else {
		for (int i = 0; i < im.c*im.h*im.w; i++) {
			im.data[i] = (im.data[i] - min) / diff;
		}
	}
}

image* sobel_image(image im)
{
    image* sobelimg = (image*)malloc(sizeof(image) * 2);
    sobelimg[0] = make_image(im.w, im.h, 1);
    sobelimg[1] = make_image(im.w, im.h, 1);

    float* gx = convolve_image(im, make_gx_filter(), 0).data;
    float* gy = convolve_image(im, make_gy_filter(), 0).data;

    for (int i = 0; i < im.h*im.w; i++) {
		sobelimg[0].data[i] = sqrtf(pow(gx[i], 2) + pow(gy[i], 2));
        sobelimg[1].data[i] = atan2(gy[i],gx[i]);
	}
    return sobelimg;
}

image colorize_sobel(image im)
{
	image* s = sobel_image(im);
    feature_normalize(s[0]);
    feature_normalize(s[1]);

    image r = make_image(im.w, im.h, 3);
    memcpy(r.data, s[1].data, im.w*im.h*sizeof(float));
    memcpy(r.data+im.w*im.h, s[0].data, im.w*im.h*sizeof(float));
    memcpy(r.data+2*im.w*im.h, s[0].data, im.w*im.h*sizeof(float));
    free_image(s[0]);
    free_image(s[1]);
    hsv_to_rgb(r);
  	return r;
}