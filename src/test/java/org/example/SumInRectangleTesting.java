package org.example;

import org.junit.jupiter.api.Assertions;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;

public class SumInRectangleTesting {
    @ParameterizedTest
    @MethodSource("cases")
    public void testSumming(int i1, int j1, int i2, int j2, String path) throws IOException {
        ImageProcessing imageProcessing = new ImageProcessing(path, BufferedImage.TYPE_BYTE_GRAY).buildIntegralImage();

        BufferedImage image = ImageIO.read(new File(path));

        Vector[][] pixels = new Vector[image.getWidth()][image.getHeight()];

        for(int i=0;i<image.getWidth();i++){
            for(int j=0;j<image.getHeight();j++){
                Color color = new Color(image.getRGB(i, j));

                pixels[i][j] = new Vector(color.getRed(), color.getGreen(), color.getBlue());
            }
        }

        float value = imageProcessing.totalInRect(j1, i1, j2, i2);

        float expected = 0;
        for(int i=i1;i<=i2;i++){
            for(int j=j1;j<=j2;j++){
                expected += pixels[i][j].x(0);
            }
        }

        Assertions.assertEquals(expected, value);
    }

    static Stream<Arguments> cases(){
        return Stream.of(
                Arguments.of(0, 0, 3, 4, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0000.png"),
                Arguments.of(2, 3, 3, 4, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0000.png"),
                Arguments.of(5, 6, 11, 23, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0001.png"),
                Arguments.of(10, 20, 14, 23, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0100.png"),
                Arguments.of(3, 10, 9, 15, "D:\\Source code\\Data\\natural_images\\person\\gray\\gray_0020.png")
        );
    }
}
