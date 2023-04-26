package org.example;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Vector{
    private final float[] points;
    Vector(float... points){
        this.points = points;
    }

    Vector(List<Float> points){
        this.points = new float[points.size()];

        for(int i=0;i<this.points.length;i++){
            this.points[i] = points.get(i);
        }
    }

    Vector(int size){
        points = new float[size];
    }

    public float x(int i){
        return points[i];
    }

    public void setX(int pos, float value){
        points[pos] = value;
    }

    public int size(){
        return points.length;
    }

    public int intRGB(){
        Color color = new Color(Math.round(points[0]), Math.round(points[1]), Math.round(points[2]));

        return color.getRGB();
    }
}

class ImageProcessing {
    private final int type;
    private final Vector[][] pixels;
    private float[][] integral;
    public ImageProcessing(String path, int type) throws IOException {
        BufferedImage image = ImageIO.read(new File(path));

        pixels = new Vector[image.getWidth()][image.getHeight()];

        for(int i=0;i<image.getWidth();i++){
            for(int j=0;j<image.getHeight();j++){
                Color color = new Color(image.getRGB(i, j));

                pixels[i][j] = new Vector(color.getRed(), color.getGreen(), color.getBlue());
            }
        }

        this.type = type;
    }

    public ImageProcessing(BufferedImage image, int type){
        pixels = new Vector[image.getWidth()][image.getHeight()];

        for(int i=0;i<image.getWidth();i++){
            for(int j=0;j<image.getHeight();j++){
                Color color = new Color(image.getRGB(i, j));

                pixels[i][j] = new Vector(color.getRed(), color.getGreen(), color.getBlue());
            }
        }

        this.type = type;
    }

    public void grayscale(String savePath) throws IOException {
        BufferedImage image = reconstruct();

        ImageFilter filter = new GrayFilter(true, 10);
        ImageProducer producer = new FilteredImageSource(image.getSource(), filter);
        Image grayImage = Toolkit.getDefaultToolkit().createImage(producer);

        BufferedImage bufferedImage= new BufferedImage(
                grayImage.getWidth(null), grayImage.getHeight(null), BufferedImage.TYPE_BYTE_GRAY);
        bufferedImage.getGraphics().drawImage(grayImage, 0, 0, null);

        ImageIO.write(bufferedImage, "png", new File(savePath));
    }

    public ImageProcessing resize(int width, int height) {
        BufferedImage image = new BufferedImage(pixels.length, pixels[0].length, type); // reconstruct the image

        for(int i=0;i<pixels.length;i++){
            for(int j=0;j<pixels[0].length;j++){
                image.setRGB(i, j, pixels[i][j].intRGB());
            }
        }

        BufferedImage resized = new BufferedImage(width, height, type);
        Graphics2D graphics = resized.createGraphics();
        graphics.drawImage(image, 0, 0, width, height, null);
        graphics.dispose();

        return new ImageProcessing(resized, type);
    }

    public ImageProcessing buildIntegralImage() { // assuming this is a grayscale image
        integral = new float[pixels.length][pixels[0].length];

        // H(i, j) = H(i - 1, j) + H(i, j - 1) - H(i - 1,j - 1) + I(i, j)
        for(int i=0;i<pixels.length;i++) {
            for(int j=0;j<pixels[0].length;j++) {
                integral[i][j] = H(i - 1, j) + H(i, j - 1) - H(i - 1,j - 1) + pixels[i][j].x(0);
            }
        }

        return this;
    }

    public Vector computeHaarFeatures() {
        // Haar features:
        // 2n x m with n: 1 -> 12 and m: 1 -> 24
        // n x 2m with n: 1 -> 24 and m: 1 -> 12
        // 2n x 2m with n: 1 -> 12 and m: 1 -> 12
        // (middle + 2n) x m where left > 0 n: 1 -> 11 and m:1 -> 24

        List<Float> features = new ArrayList<>();

        features.addAll(features(12, 24, (i, j, width, height) ->
                totalInRect(j, i, j + width - 1, i + height - 1) -
                        totalInRect(j + width, i, j + 2 * width - 1, i + height - 1)));

        features.addAll(features( 24, 12, (i, j, width, height) ->
                totalInRect(j, i, j + width - 1, i + height - 1) -
                        totalInRect(j, i + height, j + width - 1, i + 2 * height - 1)));

        features.addAll(features(12, 12, (i, j, width, height) ->
                totalInRect(j, i, j + width - 1, i + height - 1)
                        + totalInRect(j + width, i + height, j + 2 * width - 1, i + 2 * height - 1) -
                        totalInRect(j + width, i, j + 2 * width - 1, i + height - 1) -
                        totalInRect(j, i + height, j + width - 1, i + 2 * height - 1)));

        features.addAll(line());

        return new Vector(features);
    }

    private interface OnExtractingFeature {
        float extract(int x1, int y1, int width, int height) throws Exception;

    }

    private List<Float> features(int nBound, int mBound, OnExtractingFeature extractor) {
        List<Float> features = new ArrayList<>();

        for(int n=3;n<=nBound;n++) {
            for(int m=3;m<=mBound;m++) {
                for(int i=0;i<pixels.length;i++) {
                    for(int j=0;j<pixels[0].length;j++) {
                        try {
                            features.add(extractor.extract(i, j, n, m));
                        } catch (Exception e) {
                            return features;
                        }
                    }
                }
            }
        }

        return features;
    }

    private List<Float> line(){
        List<Float> features = new ArrayList<>();

        for(int n=3;n<=11;n++) {
            for(int middle=3;middle<=24 - 2*n;middle++) {
                for(int m=3;m<=24;m++){
                    for(int i=0;i<=pixels.length - m;i++) {
                        for(int j=0;j<=pixels[0].length - 2 * n - middle;j++) {
                            float left = totalInRect(j, i, j + n - 1, i + m - 1);
                            float mid = totalInRect(j + n, i, j + n + middle - 1, i + m - 1);
                            float right = totalInRect(j + middle + n, i, j + middle +  2 * n - 1, i + m - 1);

                            features.add(left + right - mid);
                        }
                    }
                }
            }
        }

        return features;
    }

    public float totalInRect(int left, int top, int right, int bottom) {
        float rightBottom = H(bottom, right);
        float leftOut = H(bottom, left - 1);
        float topOut = H(top - 1, right);
        float diagonal = H(top - 1, left - 1);

        return rightBottom - leftOut - topOut + diagonal;
    }

    public float[][] getIntegral() {
        return integral;
    }

    private float H(int i, int j) throws ArrayIndexOutOfBoundsException {
        if(i < 0 || j < 0 || i == pixels.length || j == pixels[0].length){
            return 0;
        }

        return integral[i][j];
    }

    private BufferedImage reconstruct() {
        BufferedImage image = new BufferedImage(pixels.length, pixels[0].length, type);

        for(int i=0;i<pixels.length;i++){
            for(int j=0;j<pixels[0].length;j++){
                image.setRGB(i, j, pixels[i][j].intRGB());
            }
        }

        return image;
    }
}