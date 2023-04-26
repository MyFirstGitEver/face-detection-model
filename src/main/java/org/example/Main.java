package org.example;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Stack;

public class Main {
    static String getId(int i){
        StringBuilder builder = new StringBuilder();

        int first = i % 10;
        i /= 10;
        int second = i % 10;
        i /= 10;
        int third = i % 10;
        i /= 10;
        int fourth = i % 10;

        builder.append(fourth);
        builder.append(third);
        builder.append(second);
        builder.append(first);

        return builder.toString();
    }

    public static void main(String[] args) throws IOException {
        int personCount = 600, nonPersonCount = 300;

        Pair<Vector, Float>[] dataset = new Pair[personCount + nonPersonCount];

        for(int i=0;i<personCount;i++){
            ImageProcessing processing = new ImageProcessing(
                    "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_" + getId(i) + ".png",
                    BufferedImage.TYPE_BYTE_GRAY);

            processing.buildIntegralImage();
            dataset[i] = new Pair<>(processing.computeHaarFeatures(), 1.0f);
        }

        for(int i=personCount;i<personCount + nonPersonCount;i++){
            ImageProcessing processing = new ImageProcessing(
                    "D:\\Source code\\Data\\natural_images\\non-face\\gray2\\gray_" + (i - personCount) + ".png",
                    BufferedImage.TYPE_BYTE_GRAY);

            processing.buildIntegralImage();
            dataset[i] = new Pair<>(processing.computeHaarFeatures(), -1.0f);
        }

        AdaBoost model = new AdaBoost(dataset);
        model.train(50);

        test(model, personCount, nonPersonCount);
        //nonFace();
    }

    private static void test(AdaBoost model, int personCount, int nonPersonCount) throws IOException {
        int hit = 0;

        for(int i=personCount;i<=985;i++) {
            ImageProcessing processing = new ImageProcessing(
                    "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_" + getId(i) + ".png",
                    BufferedImage.TYPE_BYTE_GRAY);

            processing.buildIntegralImage();
            if(model.isPositive(processing.computeHaarFeatures())){
                hit++;
            }
        }

        for(int i=nonPersonCount;i<=491;i++) {
            ImageProcessing processing = new ImageProcessing(
                    "D:\\Source code\\Data\\natural_images\\non-face\\gray2\\gray_" + i + ".png",
                    BufferedImage.TYPE_BYTE_GRAY);

            processing.buildIntegralImage();
            if(!model.isPositive(processing.computeHaarFeatures())){
                hit++;
            }
        }

        System.out.println(((float) hit / (985 - personCount + 491 - nonPersonCount + 2) * 100 + " %"));
    }

    private static void nonFace() throws IOException {
        int index = 0;

        for(int i=0;i<=245;i++){
            String id = getId(i);

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\motorbike\\motorbike_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\non-face\\gray2\\gray_" + index + ".png");

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\flower\\flower_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\non-face\\gray2\\gray_" + (index + 1) + ".png");

            index += 2;
        }
    }
}