package sandbox.objectdetector;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import javax.imageio.ImageIO;
import javax.swing.*;

import static sandbox.objectdetector.Utils.constructAndExecuteGraphToNormalizeImage;

public class App extends JFrame {

    public App(int width, int height) {
        setSize(width, height);
        setResizable(false);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> parsedArgs = parseArgs(args);
        String modelFolder = parsedArgs.get("model");
        String imagePath = parsedArgs.get("image");
        String labelsPath = parsedArgs.get("labels");

        if (Objects.isNull(modelFolder) || Objects.isNull(imagePath)) {
            System.err.println("Usage:"
                    + "\n --image <arg>     image file path"
                    + "\n --model <arg>     tensorflow model folder path"
                    + "\n --labels <arg>    labels file path (optional)"
                    + "\n --gui             run gui (optional)"
            );
            System.exit(1);
        }

        final SavedModelBundle modelBundle = SavedModelBundle.load(modelFolder,"serve");
        byte[] imageBytes = Files.readAllBytes(Paths.get(imagePath));
        Tensor<Float> imageInput = constructAndExecuteGraphToNormalizeImage(imageBytes);
        List<Tensor<?>> outputs = predict(modelBundle, imageInput);

        //https://github.com/fizyr/keras-retinanet#Testing
        //order is based on sequence of output tensors
        float[][] boxes = getBoxes(outputs.get(0));
        float[] scores = getScores(outputs.get(1));
        String[] labels = normalizeLabels(getLabels(outputs.get(2)), labelsPath);

        printResults(boxes, scores, labels);

        if (parsedArgs.containsKey("gui")) {
            BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageBytes));
            App gui = new App(image.getWidth(), image.getHeight());
            gui.add(new Highlighter(image, boxes, scores, labels));
            gui.setVisible(true);
        }
    }

    private static void printResults(float[][] boxes, float[] scores, String[] labels) {
        System.out.println("Print results: ");
        for (int i = 0; i < scores.length; i++) {
            float score = scores[i];
            float[] box = boxes[i];
            String label = labels[i];
            if (scores[i] > 0) {
                System.out.printf("     label: %s, score: %f, box: %s\n",
                        label, score, Arrays.toString(box));
            }
        }
    }

    private static String[] normalizeLabels(int[] rawLabels, String labelsPath) {
        String[] normalized = new String[rawLabels.length];
        String[] resolved = new String[0];
        try {
            Path path = Paths.get(labelsPath);
            resolved = Files.lines(path).toArray(String[]::new);
        } catch (Exception e) {
            System.out.println("Can't read labels file, will be used raw values");
            e.printStackTrace();
        }

        for (int i = 0; i < rawLabels.length; i++) {
            int rawLabel = rawLabels[i];
            if (rawLabel <= resolved.length - 1 && rawLabel >= 0) {
                normalized[i] = resolved[rawLabel];
            } else {
                normalized[i] = String.valueOf(rawLabel);
            }
        }

        return normalized;
    }

    private static List<Tensor<?>> predict(SavedModelBundle modelBundle, Tensor<?> input) {
        return modelBundle.session().runner().feed("input_1", input)
                //output tensors
                .fetch("filtered_detections/map/TensorArrayStack/TensorArrayGatherV3")
                .fetch("filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3")
                .fetch("filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3")
                .run();
    }

    private static float[][] getBoxes(Tensor<?> tensor) {
        int total = (int) tensor.shape()[1];
        int boxDimension = (int) tensor.shape()[2];
        return tensor.copyTo(new float[1][total][boxDimension])[0];
    }

    private static float[] getScores(Tensor<?> tensor) {
        int total = (int) tensor.shape()[1];
        return tensor.copyTo(new float[1][total])[0];
    }

    private static int[] getLabels(Tensor<?> tensor) {
        int total = (int) tensor.shape()[1];
        return tensor.copyTo(new int[1][total])[0];
    }

    private static class Highlighter extends JPanel {

        private final Image image;
        private final float[] scores;
        private final float[][] boxes;
        private final String[] labels;

        public Highlighter(Image image,  float[][] boxes, float[] scores, String[] labels) {
            this.image = image;
            this.scores = scores;
            this.boxes = boxes;
            this.labels = labels;
        }
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(image, 0, 0, null);
            g.setColor(Color.RED);

            for (int i = 0; i < scores.length; i++) {
                float score = scores[i];
                if (score < 0) {
                    break;
                }

                float[] box = boxes[i];
                int x1 = (int)box[0];
                int y1 = (int)box[1];
                int x2 = (int)box[2];
                int y2 = (int)box[3];
                g.setFont(new Font("default", Font.BOLD, 16));
                g.drawRect(x1, y1, x2 - x1, y2 - y1);
                g.drawString(String.valueOf(score), x2 + 5, y2 + 10);
                g.drawString(labels[i], x2 + 5, y1 - 5);
            }
        }
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> parsed = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i].replace("--", "");
            String value = null;
            if (i + 1 < args.length) {
                value = args[i + 1];
            }
            parsed.put(arg, value);
        }
        return parsed;
    }
}
