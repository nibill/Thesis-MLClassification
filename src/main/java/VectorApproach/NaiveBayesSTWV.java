package VectorApproach;

import Helper.UsedClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.TextDirectoryLoader;
import weka.core.pmml.jaxbbindings.SupportVectorMachine;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;


import java.awt.*;
import java.io.*;
import java.util.Random;

public class NaiveBayesSTWV
{
    /**
     *
     * @param directoryPath
     * @throws Exception
     */
    public static void DoNBSTWVEval(String directoryPath, String arffFilePath, UsedClassifier usedClassifier, boolean tfidf) throws Exception
    {
        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(new File(directoryPath));
        Instances dataSet = loader.getDataSet();

        SaveArff(arffFilePath + File.separator + "beforeFilter.arff", dataSet);

        // convert loaded text files into a word vector
        StringToWordVector filter = new StringToWordVector();

        // specify whether you want TF-IDF or not
        if(tfidf)
        {
            filter.setTFTransform(true);
            filter.setIDFTransform(true);
        }
        else
        {
            filter.setTFTransform(false);
            filter.setIDFTransform(false);
        }

        filter.setInputFormat(dataSet);
        dataSet = Filter.useFilter(dataSet, filter);

        SaveArff(arffFilePath + File.separator + "afterFilter.arff", dataSet);

        Evaluation eval = new Evaluation(dataSet);

        if(usedClassifier == UsedClassifier.NB)
        {
            NaiveBayes naiveBayes = new NaiveBayes();
            eval.crossValidateModel(naiveBayes, dataSet, 10, new Random(1));
        }
        else if(usedClassifier == UsedClassifier.MNNB)
        {
            NaiveBayesMultinomial naiveBayesMultinomial = new NaiveBayesMultinomial();
            eval.crossValidateModel(naiveBayesMultinomial, dataSet, 10, new Random(1));

            /*ThresholdCurve tc = new ThresholdCurve();
            int classIndex = 0;
            Instances result = tc.getCurve(eval.predictions(), classIndex);

            ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
            vmc.setROCString("(Area under ROC = "
                + Utils.doubleToString(tc.getROCArea(result), 4) + ")");
            vmc.setName(result.relationName());
            PlotData2D tempd = new PlotData2D(result);
            tempd.setPlotName(result.relationName());
            tempd.addInstanceNumberAttribute();
            vmc.addPlot(tempd);

            // display curve
            String plotName = vmc.getName();
            final javax.swing.JFrame jf = new javax.swing.JFrame(
                "Weka Classifier Visualize: " + plotName);
            jf.setSize(500, 400);
            jf.getContentPane().setLayout(new BorderLayout());
            jf.getContentPane().add(vmc, BorderLayout.CENTER);
            jf.addWindowListener(new java.awt.event.WindowAdapter() {
                public void windowClosing(java.awt.event.WindowEvent e) {
                    jf.dispose();
                }
            });
            jf.setVisible(true);*/
        }

        System.out.println(eval.toSummaryString("\nResults\n======\n", true));

        System.out.println("\nCorrect: " + eval.toMatrixString());
    }

    private static void SaveArff(String path, Instances data) throws Exception
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        writer.write(data.toString());
        writer.flush();
        writer.close();
    }
}

