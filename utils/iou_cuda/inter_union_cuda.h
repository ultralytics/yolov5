/* reference: https://github.com/NVIDIA/retinanet-examples */

#pragma once


/* Below functions are used for iou computation (polygon);
Boxes have shape nx8 and Anchors have mx8;
Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
*/
int inter_union(
    const void *const *inputs, void *const *outputs,
    int num_boxes, int num_anchors, cudaStream_t stream);


/* Below functions are used for loss computation (polygon);
For boxes and anchors having the same shape: nx8;
Return intersection and union of boxes[i, :] and anchors[i, :] with shape of (n, ).
*/
int b_inter_union(
    const void *const *inputs, void *const *outputs,
    int num_anchors, cudaStream_t stream);
