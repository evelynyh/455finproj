// Copyright 2024 Evelyn Heckman

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c) {
    // clamp
    y = fmin(im.h - 1, y);
    x = fmin(im.w - 1, x);
    c = fmin(im.c - 1, c);
    y = fmax(0, y);
    x = fmax(0, x);
    c = fmax(0, c);

    float* ptr = im.data;
    ptr += (im.h * im.w * c);  // get to proper channel
    ptr += (im.w * y);  // get to proper row
    ptr += x;  // get to proper column
    return *ptr;
}

void set_pixel(image im, int x, int y, int c, float v) {
    // clamp
    y = fmin(im.h - 1, y);
    x = fmin(im.w - 1, x);
    c = fmin(im.c - 1, c);
    y = fmax(0, y);
    x = fmax(0, x);
    c = fmax(0, c);


    float* ptr = im.data;
    ptr += (im.h * im.w * c);  // get to proper channel
    ptr += (im.w * y);  // get to proper row
    ptr += x;  // get to proper column
    *ptr = v;  // set value
}

image copy_image(image im) {
    image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, ((sizeof(float)) * im.w * im.h * im.c));
    return copy;
}

image rgb_to_grayscale(image im) {
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    float* ptr_g = gray.data;
    float* ptr_i = im.data;
    for (int i = 0; i < gray.h * gray.w; i++) {
        *ptr_g = 0.299 * (*ptr_i) + 0.587 * (*(ptr_i + gray.h * gray.w)) +
            .114 * (*(ptr_i + gray.h * gray.w * 2));
        ptr_g++;
        ptr_i++;
    }
    clamp_image(gray);
    return gray;
}

void shift_image(image im, int c, float v) {
    float* ptr = im.data;
    c = fmin(im.c - 1, c);
    c = fmax(0, c);

    ptr += im.h * im.w * c;  // get to channel
    for (int i = 0; i < im.h * im.w; i++) {
        *ptr = *ptr + v;
        ptr++;
    }
    clamp_image(im);
}

void clamp_image(image im) {
    float* ptr = im.data;
    for (int i = 0; i < im.h * im.w * im.c; i++) {
        *ptr = fmin(1, *ptr);
        *ptr = fmax(0, *ptr);
        ptr++;
    }
}

// These might be handy
float three_way_max(float a, float b, float c) {
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c);
}

float three_way_min(float a, float b, float c) {
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c);
}

void rgb_to_hsv(image im) {
    // edge case: grayscale
    if (im.c == 0) {
        float* ptr = im.data;
        image copy = make_image(im.w, im.h, 3);
        float* h = copy.data;
        float* s = h + im.w * im.h;
        float* v = s + im.w * im.h;
        for (int i = 0; i < im.h * im.w; i++) {
            *h = 0;
            *s = *ptr;
            *v = *ptr;

            // increment!
            h++;
            s++;
            v++;
            ptr++;
        }
        return;
    }

    float* h = im.data;
    float* s = h + im.h * im.w;
    float* v = s + im.h * im.w;
    for (int i = 0; i < im.h * im.w; i++) {
        float v_upper = three_way_max(*h, *s, *v);
        float m = three_way_min(*h, *s, *v);
        float c = v_upper - m;
        float h_prime = 0;
        if (c == 0) {
            h_prime = 0;
        } else if (v_upper == *h) {  // red
            h_prime = (*s - *v) / c;
        } else if (v_upper == *s) {  // green
            h_prime = (*v - *h) / c + 2;
        } else if (v_upper == *v) {  // blue
            h_prime = (*h - *s) / c + 4;
        }

        // set values
        if (h_prime < 0) {
            *h = h_prime / 6 + 1;
        } else {
            *h = h_prime / 6;
        }
        if (v_upper == 0) {
            *s = 0;
        } else {
            *s = c / v_upper;
        }
        *v = v_upper;

        // increment!
        h++;
        s++;
        v++;
    }
}

void hsv_to_rgb(image im) {
    assert(im.c == 3);
    float* r = im.data;
    float* g = r + im.h * im.w;
    float* b = g + im.h * im.w;

    for (int i = 0; i < im.h * im.w; i++) {
        float h = (*r) * 6;
        float hi = floor(h);
        float f = h - hi;
        float p = (*b) * (1 - (*g));
        float q = (*b) * (1 - f * (*g));
        float t = (*b) * (1 - (1 - f) * (*g));

        // set values
        if (hi == 0) {
            *r = *b;
            *g = t;
            *b = p;
        } else if (hi == 1) {
            *r = q;
            *g = *b;
            *b = p;
        } else if (hi == 2) {
            *r = p;
            *g = *b;
            *b = t;
        } else if (hi == 3) {
            *r = p;
            *g = q;
        } else if (hi == 4) {
            *r = t;
            *g = p;
        } else {
            *r = *b;
            *g = p;
            *b = q;
        }

        // increment!
        r++;
        g++;
        b++;
    }
}
