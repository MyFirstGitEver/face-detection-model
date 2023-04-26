package org.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

class Classifier {
    public Classifier(float weight, int index) {
        this.weight = weight;
        this.index = index;
    }

    float weight;
    int index;
}

public class AdaBoost{
    private Pair<Vector, Float>[] dataset;
    private int positiveCount, negativeCount;
    private float[] thresholds;
    private List<Classifier> classifiers;
    AdaBoost(Pair<Vector, Float>[] dataset) {
        this.dataset = dataset;
        this.thresholds = new float[dataset[0].first.size()];
        this.classifiers = new ArrayList<>();

        for(Pair<Vector, Float> point : dataset){
            if(point.second == -1){
                negativeCount++;
            }
            else{
                positiveCount++;
            }
        }

        for(int i=0;i<thresholds.length;i++) {
            thresholds[i] = getThreshold(i);
        }
    }

    private float getThreshold(int i) {
        Arrays.sort(dataset, new Comparator<Pair<Vector, Float>>() {
            @Override
            public int compare(Pair<Vector, Float> o1, Pair<Vector, Float> o2) {
                return Float.compare(o1.first.x(i), o2.first.x(i));
            }
        });

        int positive = 0, negative = 0;

        int choice = 0;
        float currentGini = Float.MAX_VALUE;
        for(int j=0;j<dataset.length;j++) {
            if(dataset[j].second == -1){
                negative++;
            }
            else{
                positive++;
            }

            float giniLeft = gini(positive, negative);
            float giniRight = gini(positiveCount - positive, negativeCount - negative);

            float leftPercent = (float) (j + 1) / dataset.length;
            float gini = giniLeft * leftPercent + (1 - leftPercent) * giniRight;

            if(gini < currentGini){
                currentGini = gini;
                choice = j;
            }
        }

        return dataset[choice].first.x(i);
    }

    public void train(int num) {
        for(int iteration=0;iteration<num;iteration++) {
            float totalWeight = 0;

            for(Pair<Vector, Float> p : dataset) {
                totalWeight += Math.exp(-p.second * w(p.first));
            }

            int index = 0;
            float min = Float.MAX_VALUE;
            for(int i=0;i<thresholds.length;i++) {
                float minError = 0;

                for(Pair<Vector, Float> p : dataset) {
                    int temp = 1;
                    if(p.first.x(i) <= thresholds[i]){
                        temp = -1;
                    }

                    if(temp * p.second < 0) {
                        minError += Math.exp(-p.second * w(p.first));
                    }
                }

                if(min > minError){
                    min = minError;
                    index = i;
                }
            }

            min = min / totalWeight + 0.000001f;
            double alpha = Math.log((1 - min) / min) / 2;

            if(Double.isNaN(alpha)){
                return;
            }

            classifiers.add(new Classifier((float) alpha, index));
        }
    }

    public boolean isPositive(Vector v) {
        if(v.size() != dataset[0].first.size()){
            return false;
        }

        return w(v) >= 0;
    }

    public float error() {
        if(dataset.length == 0){
            return Float.MAX_VALUE;
        }

        float error = 0;

        for(Pair<Vector, Float> p : dataset) {
            float predInPercent = w(p.first);
            error += Math.exp(-p.second * predInPercent);
        }

        return error;
    }

    private float w(Vector v){
        if(classifiers.size() == 0) {
            return 1;
        }

        float pred = 0;

        for(Classifier classifier : classifiers){
            int temp = v.x(classifier.index) <= thresholds[classifier.index] ? -1 : 1;

            pred += temp * classifier.weight;
        }

        return pred;
    }

    private float gini(int yes, int no) {
        float term1 = (float) yes / (yes + no);
        float term2 = (float) no / (yes + no);

        return (1.0f - term1 * term1 - term2 * term2);
    }
}