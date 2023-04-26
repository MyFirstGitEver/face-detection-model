package org.example;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;

public class HaarFeaturesTesting {
    Vector[][] pixels;
    float[][] integral;

    public static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of("D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0000.png"),
                Arguments.of("D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0002.png"),
                Arguments.of("D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0003.png"),
                Arguments.of("D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0100.png"),
                Arguments.of("D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0150.png"),
                Arguments.of("D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0200.png"),
                Arguments.of("D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0101.png")
        );
    }

    private interface OnExtractingFeature {
        float extract(int x1, int y1, int width, int height) throws Exception;
    }

    private void loadPixels(String path) throws IOException {
        BufferedImage image = ImageIO.read(new File(path));

        pixels = new Vector[image.getWidth()][image.getHeight()];

        for(int i=0;i<image.getWidth();i++){
            for(int j=0;j<image.getHeight();j++){
                Color color = new Color(image.getRGB(i, j));

                pixels[i][j] = new Vector(color.getRed(), color.getGreen(), color.getBlue());
            }
        }
    }

    @ParameterizedTest
    @MethodSource("cases")
    public void haarTest(String path) throws IOException {
        loadPixels(path);
        buildIntegralImage();

        features(0, 12, 24, (i, j, width, height) ->
                totalInRect(j, i, j + width - 1, i + height - 1) -
                        totalInRect(j + width, i, j + 2 * width - 1, i + height - 1));

        features(1, 24, 12, (i, j, width, height) ->
                totalInRect(j, i, j + width - 1, i + height - 1) -
                        totalInRect(j, i + height, j + width - 1, i + 2 * height - 1));

        features(2, 12, 12, (i, j, width, height) ->
                totalInRect(j, i, j + width - 1, i + height - 1)
                        + totalInRect(j + width, i + height, j + 2 * width - 1, i + 2 * height - 1) -
                        totalInRect(j + width, i, j + 2 * width - 1, i + height - 1) -
                        totalInRect(j, i + height, j + width - 1, i + 2 * height - 1));

        line();
    }

    private void features(int type, int nBound, int mBound, OnExtractingFeature extractor) { // 2n x m
        for(int n=3;n<=nBound;n++) {
            for(int m=3;m<=mBound;m++) {
                for(int i=0;i<pixels.length;i++) {
                    for(int j=0;j<pixels[0].length;j++) {
                        try {
                            Assertions.assertEquals(test(type, i, j, m, n, -1), extractor.extract(i, j, n, m));
                        } catch (Exception e) {
                            return;
                        }
                    }
                }
            }
        }
    }

    private void line(){
        for(int n=3;n<=11;n++) {
            for(int middle=3;middle<=24 - 2*n;middle++) {
                for(int m=3;m<=24;m++){
                    for(int i=0;i<=pixels.length - m;i++) {
                        for(int j=0;j<=pixels[0].length - 2 * n - middle;j++) {
                            float left = totalInRect(j, i, j + n - 1, i + m - 1);
                            float mid = totalInRect(j + n, i, j + n + middle - 1, i + m - 1);
                            float right = totalInRect(j + middle + n, i, j + middle +  2 * n - 1, i + m - 1);

                            Assertions.assertEquals(test(3, i, j, n, m, middle), left + right - mid);
                        }
                    }
                }
            }
        }
    }

    public float totalInRect(int left, int top, int right, int bottom) {
        float rightBottom = H(bottom, right);
        float leftOut = H(bottom, left - 1);
        float topOut = H(top - 1, right);
        float diagonal = H(top - 1, left - 1);

        return rightBottom - leftOut - topOut + diagonal;
    }

    private float H(int i, int j) throws ArrayIndexOutOfBoundsException {
        if(i < 0 || j < 0 || i == pixels.length || j == pixels[0].length){
            return 0;
        }

        return integral[i][j];
    }

    public void buildIntegralImage() { // assuming this is a grayscale image
        integral = new float[pixels.length][pixels[0].length];

        // H(i, j) = H(i - 1, j) + H(i, j - 1) - H(i - 1,j - 1) + I(i, j)
        for(int i=0;i<pixels.length;i++) {
            for(int j=0;j<pixels[0].length;j++) {
                integral[i][j] = H(i - 1, j) + H(i, j - 1) - H(i - 1,j - 1) + pixels[i][j].x(0);
            }
        }
    }

    private float test(int type, int x, int y, int width, int height, int middle) {
        float expected = 0;

        switch (type) {
            case 0 -> {
                for (int i = x; i < x + height; i++) {
                    for (int j = y; j < y + 2 * width; j++) {
                        if (j - y + 1 <= width) {
                            expected += pixels[i][j].x(0);
                        } else {
                            expected -= pixels[i][j].x(0);
                        }
                    }
                }
            }
            case 1 -> {
                for (int i = x; i < x + 2 * height; i++) {
                    for (int j = y; j < y + width; j++) {
                        if (i - x + 1 <= height) {
                            expected += pixels[i][j].x(0);
                        } else {
                            expected -= pixels[i][j].x(0);
                        }
                    }
                }
            }
            case 2 -> {
                for (int i = x; i < x + 2 * height; i++) {
                    for (int j = y; j < y + 2 * width; j++) {
                        if (((i - x + 1) <= height && (j - y + 1) <= width) ||
                                ((i - x + 1) > height && (j - y + 1) > width)) {
                            expected += pixels[i][j].x(0);
                        } else {
                            expected -= pixels[i][j].x(0);
                        }
                    }
                }
            }
            default -> {
                for (int i = x; i < x + height; i++) {
                    for (int j = y; j < y + 2 * width + middle; j++) {
                        if ((j - y + 1) > width && (j - y + 1) <= width + middle) {
                            expected -= pixels[i][j].x(0);
                        } else {
                            expected += pixels[i][j].x(0);
                        }
                    }
                }
            }
        }

        return expected;
    }
}
