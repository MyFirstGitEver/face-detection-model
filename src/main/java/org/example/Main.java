package org.example;

import java.awt.image.BufferedImage;
import java.io.IOException;

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
        int personCount = 700, nonPersonCount = 738;

        Pair<Vector, Float>[] dataset = new Pair[personCount + nonPersonCount];

        for(int i=0;i<personCount;i++){
            ImageProcessing processing = new ImageProcessing(
                    "D:\\Source code\\Data\\natural_images\\grays\\gray\\gray_" + i + ".png",
                    BufferedImage.TYPE_BYTE_GRAY);

            processing.buildIntegralImage();
            dataset[i] = new Pair<>(processing.computeHaarFeatures(), 1.0f);

            if(i % 100 == 0) {
                System.out.println(i + " person images processed");
            }
        }

        for(int i=personCount;i<personCount + nonPersonCount;i++){
            ImageProcessing processing = new ImageProcessing(
                    "D:\\Source code\\Data\\natural_images\\grays\\gray2\\gray_" +  (i - personCount) + ".png",
                    BufferedImage.TYPE_BYTE_GRAY);

            processing.buildIntegralImage();
            dataset[i] = new Pair<>(processing.computeHaarFeatures(), -1.0f);

            if(i % 100 == 0) {
                System.out.println((i - personCount) + " non-face images processed");
            }
        }

        AdaBoost model = new AdaBoost(dataset);
        //model.train(1000);
        test(model);
        //buildTrainingSet();
        //buildTestSet();
    }

    private static void test(AdaBoost model) throws IOException {
        int hit = 0;
        double total = 0;

        for(int i=0;i<465;i++) {
            ImageProcessing processing = new ImageProcessing(
                    "D:\\Source code\\Data\\natural_images\\grays\\test2\\gray_" + i + ".png",
                    BufferedImage.TYPE_BYTE_GRAY);

            processing.buildIntegralImage();
            if(!model.isPositive(processing.computeHaarFeatures())){
                hit++;
            }

            total++;
        }

        for(int i=465;i<=750;i++) {
            ImageProcessing processing = new ImageProcessing(
                    "D:\\Source code\\Data\\natural_images\\grays\\test\\gray_" + i + ".png",
                    BufferedImage.TYPE_BYTE_GRAY);

            processing.buildIntegralImage();
            if(model.isPositive(processing.computeHaarFeatures())){
                hit++;
            }

            total++;
        }

        System.out.println(hit / total * 100 + " %");
    }

    private static void buildTestSet() throws IOException {
        int index = 0;

        for(int i=246;i<=400;i++) {
            String id = getId(i);

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\motorbike\\motorbike_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\grays\\test2\\gray_" + index + ".png");

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\flower\\flower_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\grays\\test2\\gray_" + (index + 1) + ".png");

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\fruit\\fruit_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\grays\\test2\\gray_" + (index + 2) + ".png");

            index += 3;
        }

        for(int i=700;i<986;i++) {
            String id = getId(i);

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\person\\person_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\grays\\test\\gray_" + index + ".png");
            index++;
        }
    }

    private static void buildTrainingSet() throws IOException {
        int index = 0;

        for(int i=0;i<=245;i++) {
            String id = getId(i);

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\motorbike\\motorbike_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\grays\\gray2\\gray_" + index + ".png");

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\flower\\flower_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\grays\\gray2\\gray_" + (index + 1) + ".png");

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\fruit\\fruit_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\grays\\gray2\\gray_" + (index + 2) + ".png");

            index += 3;
        }

        for(int i=0;i<700;i++) {
            String id = getId(i);

            new ImageProcessing("D:\\Source code\\Data\\natural_images\\person\\person_" + id + ".jpg",
                    BufferedImage.TYPE_INT_RGB)
                    .resize(24, 24)
                    .grayscale("D:\\Source code\\Data\\natural_images\\grays\\gray\\gray_" + i + ".png");
        }
    }
}