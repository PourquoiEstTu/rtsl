import {
  HandLandmarker,
  PoseLandmarker,
  type PoseLandmarkerResult,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

export function drawHandLandmarks(
  ctx: CanvasRenderingContext2D,
  handLandmarkerResults: HandLandmarkerResult,
  W: number,
  H: number,
) {
  for (const landmarks of handLandmarkerResults.landmarks) {
    // Draw all 21 landmarks as circles
    for (const landmark of landmarks) {
      const x = landmark.x * W;
      const y = landmark.y * H;

      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "#FFFF00";
      ctx.fill();
    }

    // Draw connections between landmarks
    const connections = HandLandmarker.HAND_CONNECTIONS;
    ctx.strokeStyle = "#FFFF00";
    ctx.lineWidth = 2;

    for (const conn of connections) {
      const start = landmarks[conn.start];
      const end = landmarks[conn.end];
      if (!start || !end) continue;

      ctx.beginPath();
      ctx.moveTo(start.x * W, start.y * H);
      ctx.lineTo(end.x * W, end.y * H);
      ctx.stroke();
    }
  }
}

export function drawPoseLandmarks(
  ctx: CanvasRenderingContext2D,
  poseLandmarkerResults: PoseLandmarkerResult,
  W: number,
  H: number,
) {
  for (const landmarks of poseLandmarkerResults.landmarks) {
    // Draw all 21 landmarks as circles
    for (const landmark of landmarks) {
      const x = landmark.x * W;
      const y = landmark.y * H;

      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "#007FFF";
      ctx.fill();
    }

    // Draw connections between landmarks
    const connections = PoseLandmarker.POSE_CONNECTIONS;
    ctx.strokeStyle = "#007FFF";
    ctx.lineWidth = 2;

    for (const conn of connections) {
      const start = landmarks[conn.start];
      const end = landmarks[conn.end];
      if (!start || !end) continue;

      ctx.beginPath();
      ctx.moveTo(start.x * W, start.y * H);
      ctx.lineTo(end.x * W, end.y * H);
      ctx.stroke();
    }
  }
}
