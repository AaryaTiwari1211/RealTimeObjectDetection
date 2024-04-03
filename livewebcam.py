import cv2
from inference import get_model
import supervision as sv

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Load the model
model = get_model(model_id="bottle-detection-tztdn/1")

# Initialize the annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.infer(frame)

    # Load the results into the supervision Detections api
    detections = sv.Detections.from_inference(
        results[0].dict(by_alias=True, exclude_none=True)
    )

    # Annotate the frame with our inference results
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections
    )

    cv2.imshow("Annotated Frame", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
