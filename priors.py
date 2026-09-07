import math
import cv2
import numpy as np

import torch


def sky_seg(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    (_, v_mask) = cv2.threshold(v_channel, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(image, 100, 200)
    # Keep the original run.py convention for identical sky masks.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = cv2.GaussianBlur(sobelx, (5, 5), 0)
    sobely = cv2.GaussianBlur(sobely, (5, 5), 0)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.convertScaleAbs(sobel_edges)
    (_, sobel_edges) = cv2.threshold(sobel_edges, 5, 255, cv2.THRESH_BINARY)
    laplacian_edges = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_edges = cv2.GaussianBlur(laplacian_edges, (5, 5), 0)
    laplacian_edges = cv2.convertScaleAbs(laplacian_edges)
    (_, laplacian_edges) = cv2.threshold(laplacian_edges, 5, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_or(sobel_edges, edges)
    edges = cv2.bitwise_or(edges, laplacian_edges)
    edges_mask = cv2.bitwise_not(edges)
    final_mask = cv2.bitwise_and(v_mask, edges_mask)
    (contours, _) = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    total_area = 0
    total_perimeter = 0
    for contour in contours:
        total_area += cv2.contourArea(contour)
        total_perimeter += cv2.arcLength(contour, True)
    C = 0.035
    r = final_mask.shape[0] * final_mask.shape[1] / (512 * 512)
    kernel_size = int(r * math.sqrt(total_area / np.pi) * C)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    (contours, _) = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = 100
    for contour in contours:
        if cv2.contourArea(contour) <= min_area:
            cv2.drawContours(final_mask, [contour], -1, 0, thickness=cv2.FILLED)
    final_mask = final_mask.astype(float)
    if np.sum(final_mask) != 0:
        final_mask = (final_mask - np.min(final_mask)) / max(
            float(np.max(final_mask) - np.min(final_mask)), 1e-6
        )
    return final_mask


def DarkChannel(im, sz, sky=True):
    (b, g, r) = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel) / 255
    if sky:
        sky_mask = sky_seg(im)
        dark[sky_mask == 1] = 0.1
        C = 0.2
        total_area = np.sum(sky_mask)
        kernel_size = int(math.sqrt(total_area / np.pi) * C)
        if kernel_size % 2 == 0:
            kernel_size += 1
    return dark


def calc_mean_std_qkv(feat, eps=1e-05):
    (N, HW, C) = feat.shape
    feat_var = feat.var(dim=1) + eps
    feat_std = feat_var.sqrt().view(N, 1, C)
    feat_mean = feat.mean(dim=1).view(N, 1, C)
    return (feat_mean, feat_std)


@torch.no_grad()
def adaptive_instance_normalization_qkv(content_feat, style_feat):
    assert content_feat.size() == style_feat.size()
    size = content_feat.size()
    (style_mean, style_std) = calc_mean_std_qkv(style_feat)
    (content_mean, content_std) = calc_mean_std_qkv(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-05):
    size = feat.size()
    assert len(size) == 4
    (N, C) = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return (feat_mean, feat_std)


def adaptive_instance_normalization(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    (style_mean, style_std) = calc_mean_std(style_feat)
    (content_mean, content_std) = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def modified_sigmoid_adjusted(x, a=10):
    if abs(a) < 1e-6:
        return x
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return (torch.sigmoid(a * (x - 0.5)) - sigmoid(-a * 0.5)) / (
        sigmoid(a * 0.5) - sigmoid(-a * 0.5)
    )


def compute_mean_cov(feature, mask=None):
    (N, C, H, W) = feature.size()
    feature_flat = feature.view(N, C, -1)
    mean = torch.mean(feature_flat, dim=2, keepdim=True)
    feature_centered = feature_flat - mean
    cov = torch.matmul(feature_centered.transpose(1, 2), feature_centered) / (H * W)
    return (mean, cov)
