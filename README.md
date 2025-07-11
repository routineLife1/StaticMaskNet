# StaticMaskNet

**An approach for generating robust static region masks under challenging illumination and transparency conditions.**
> **Abstract:** Identifying static regions in videos is highly effective for preventing distortion in these areas during optical
> flow-based frame interpolation. However, regions that appear static to the human eye often exhibit subtle changes in
> transparency, illumination, or noise, making it challenging to detect them using traditional computer vision techniques
> such as OpenCV-based operations. To address this, a robust method is required to generate static masks that are
> resilient to such variations. It has been observed that some high-precision optical flow algorithms perform well in
> preserving static regions when operating at high resolutions, but computing full-resolution optical flow is
> computationally expensive. Therefore, this project proposes a novel framework that leverages high-accuracy optical flow
> to generate supervision labels, and trains a lightweight network to learn these labels. This enables the efficient
> generation of static region masks that are robust to changes in transparency, lighting, and noise, with significantly
> reduced computational cost.

# ▶️Demo

The top part of the video shows the output of the frame interpolation network without applying the mask, while the
bottom part shows the result after applying the mask.

https://github.com/user-attachments/assets/1673374f-a571-4422-b1de-afcc2679e94c

# 🚀 Model training

To train the model, modify the following files as needed:
- dataset.py – customize your dataset loading logic
- train.py – configure training parameters and start training

# 🔍 Inference

Coming soon.

# 📖Author's Note

>This project is intended as a proof-of-concept. Due to time constraints, I was unable to fully refine it. However, in the interest of advancing video frame interpolation research, I have decided to share the core idea early. If you are interested in experimenting with the pretrained weights, feel free to contact me. If you'd like to further explore this direction, the proposed network can be viewed as a **distillation framework**, where the ability to detect static regions is distilled from a high-accuracy optical flow model. Here are several possible research directions:

1. Use **dmap** generated by *Exploring-Discontinuity-for-VFI* as a proxy ground truth, and run inference on challenging dynamic datasets. Evaluate the flow-based masks by computing the L1 loss against the dmap in static regions to benchmark flow quality in these areas.
2. Simplify or improve the architecture (I believe a smaller and faster network may be sufficient).
3. Design loss functions better suited for this specific task.
4. Joint refine with vfi network.
5. Extend the input to multiple frames (B, N, H, W) and provide temporal cues such as inter-frame differences. Let the network output multiple masks (B, N−1, H, W) in one forward pass, enabling it to learn how to selectively leverage spatiotemporal context. This could potentially improve mask quality while achieving an (N−1)-fold speedup during inference.


# 🌟Acknowledgment
**This project is supported by [SVFI Development Team](https://github.com/Justin62628/Squirrel-RIFE)**

# 🔗Reference
[[Practical-RIFE]](https://github.com/hzwer/Practical-RIFE) [[Exploring-Discontinuity-for-VFI]](https://github.com/pandatimo/Exploring-Discontinuity-for-VFI)