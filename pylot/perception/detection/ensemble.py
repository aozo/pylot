from pylot.perception.detection.utils import BoundingBox2D
import statistics


def calculate_bbox_iou(bbox_1, bbox_2, img_width, img_height):
    bb1 = BoundingBox2D(
        int(bbox_1[1] * img_width),
        int(bbox_1[3] * img_width),
        int(bbox_1[0] * img_height),
        int(bbox_1[2] * img_height))

    bb2 = BoundingBox2D(
        int(bbox_2[1] * img_width),
        int(bbox_2[3] * img_width),
        int(bbox_2[0] * img_height),
        int(bbox_2[2] * img_height))

    return bb1.calculate_iou(bb2)


def calculate_avg_bbox(detections):
    ymins = []
    xmins = []
    ymaxes = []
    xmaxes = []
    for bbox, _, _ in detections:
        (ymin, xmin, ymax, xmax) = bbox
        ymins.append(ymin)
        xmins.append(xmin)
        ymaxes.append(ymax)
        xmaxes.append(xmax)
    return [statistics.mean(ymins),
            statistics.mean(xmins),
            statistics.mean(ymaxes),
            statistics.mean(xmaxes)]


def calculate_avg_score(detections):
    scores = []
    for _, score, _ in detections:
        scores.append(score)
    return statistics.mean(scores)


def calculate_ensemble(model_1_detections,
                       model_2_detections,
                       model_3_detections,
                       img_width,
                       img_height):

    # List of overlapping bbox detections
    overlap_dets = []

    for m1_det in model_1_detections:
        overlap_dets.append([m1_det])

    for m2_det in model_2_detections:
        bbox_2, score_2, cls_2 = m2_det

        found_overlap = False
        for i, dets in enumerate(overlap_dets):
            bbox, score, cls = dets[0]
            iou = calculate_bbox_iou(bbox, bbox_2, img_width, img_height)
            if (iou >= 0.5 and cls == cls_2):
                overlap_dets[i].append(m2_det)
                found_overlap = True

        # If this detection did not overlap with any existing detection, then
        # add it as a unique detection
        if not found_overlap:
            overlap_dets.append([m2_det])

    for m3_det in model_3_detections:
        bbox_3, score_3, cls_3 = m3_det

        for i, dets in enumerate(overlap_dets):
            bbox, score, cls = dets[0]
            iou = calculate_bbox_iou(bbox, bbox_3, img_width, img_height)
            if (iou >= 0.5 and cls == cls_3):
                overlap_dets[i].append(m3_det)

        # If a model 3 detection does not have sufficient bbox overlap with
        # any detection from model 1 or model 2, then it's ignored

    classes = []
    bboxes = []
    scores = []

    for dets in overlap_dets:
        if len(dets) > 1:
            _, _, maj_cls = dets[0]
            avg_bbox = calculate_avg_bbox(dets)
            avg_score = calculate_avg_score(dets)

            classes.append(maj_cls)
            bboxes.append(avg_bbox)
            scores.append(avg_score)

    num_detections = len(classes)

    return num_detections, bboxes, scores, classes
