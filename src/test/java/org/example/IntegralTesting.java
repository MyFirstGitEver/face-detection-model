package org.example;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;

public class IntegralTesting {

    @ParameterizedTest
    @MethodSource("cases")
    public void test(int size, String path) throws IOException {
        ImageProcessing imageProcessing = new ImageProcessing(path, BufferedImage.TYPE_BYTE_GRAY).buildIntegralImage();
        float[][] integral = imageProcessing.getIntegral();

        BufferedImage image = ImageIO.read(new File(path));

        Vector[][] pixels = new Vector[image.getWidth()][image.getHeight()];

        for(int i=0;i<image.getWidth();i++){
            for(int j=0;j<image.getHeight();j++){
                Color color = new Color(image.getRGB(i, j));

                pixels[i][j] = new Vector(color.getRed(), color.getGreen(), color.getBlue());
            }
        }

        for(int i=0;i<pixels.length;i++) {
            for(int j=0;j<pixels[0].length;j++) {
                float sum = 0;

                for(int a= i - size + 1;a<=i;a++){
                    for(int b= j - size + 1;b<=j;b++){
                        if(a < 0 || b < 0){
                            continue;
                        }

                        sum += pixels[a][b].x(0);
                    }
                }

                Assertions.assertEquals(sum, get(integral, i, j) -
                        get(integral, i - size, j) - get(integral, i, j - size) + get(integral, i - size, j - size));
            }
        }
    }

    private float get(float[][] integral, int i, int j){
        if(i < 0 || j < 0 || i == integral.length || j == integral[0].length){
            return 0;
        }

        return integral[i][j];
    }

    static Stream<Arguments> cases(){
        return Stream.of(
                Arguments.of(3, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0000.png"),
                Arguments.of(2, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0000.png"),
                Arguments.of(1, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0000.png"),
                Arguments.of(15, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0000.png"),
                Arguments.of(10, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0000.png"),
                Arguments.of(4, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0001.png"),
                Arguments.of(4, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0001.png"),
                Arguments.of(2, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0001.png"),
                Arguments.of(3, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0001.png")
        );
    }
}