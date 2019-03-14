package NNApproach;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Random;

/**
 * Textclassification with CNN
 *
 * @author Alex Black
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/sentenceclassification/CnnSentenceClassificationExample.java
 * @author Thomas Buchegger
 */

public class CNNClassification
{
    /**
     * Does the classification
     *
     * @param vectorFilePath Path to the pre-trained vectors
     * @param baseTrainPath  Path to train/test files
     * @param cat            Should be classified into categories or machen/schauen
     * @throws Exception
     */
    public static void CNNClassify(String vectorFilePath, String baseTrainPath, boolean cat) throws Exception
    {
        // basic configuration
        int catCounter = 6;
        int batchSize = 32;
        int vectorSize = 300;
        int nEpochs = 20;
        int truncateEventsToLength = 256; //# of words
        int cnnLayerFeatureMaps = 100;
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345);

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        if (cat)
        {
            catCounter = 6;
        } else
        {
            catCounter = 2;
        }

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
            .weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(new Adam(0.01))
            .convolutionMode(ConvolutionMode.Same)
            .l2(0.0001)
            .graphBuilder()
            .addInputs("input")
            .addLayer("cnn3", new ConvolutionLayer.Builder()
                .kernelSize(3, vectorSize)
                .stride(1, vectorSize)
                .nIn(1)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(4, vectorSize)
                .stride(1, vectorSize)
                .nIn(1)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                .kernelSize(5, vectorSize)
                .stride(1, vectorSize)
                .nIn(1)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(globalPoolingType)
                .dropOut(0.5)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(3 * cnnLayerFeatureMaps)
                .nOut(catCounter)    //classes, one for each category
                .build(), "globalPool")
            .setOutputs("out").backprop(true)
            .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("Number of parameters by layer:");
        for (Layer l : net.getLayers())
        {
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }

        //Load word vectors and get the DataSetIterators for training and testing
        System.out.println("Loading word vectors and creating DataSetIterators");
        System.out.println();
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(vectorFilePath));

        DataSetIterator trainIter;
        DataSetIterator testIter;

        // train dataset based on whether you want categories or machen/schauen
        if (cat)
        {
            trainIter = DataSetStuff.IterateDataSet(baseTrainPath, true, wordVectors, batchSize, truncateEventsToLength, rng, true);
            testIter = DataSetStuff.IterateDataSet(baseTrainPath, false, wordVectors, batchSize, truncateEventsToLength, rng, true);
        } else
        {
            trainIter = DataSetStuff.IterateDataSet(baseTrainPath, true, wordVectors, batchSize, truncateEventsToLength, rng, false);
            testIter = DataSetStuff.IterateDataSet(baseTrainPath, false, wordVectors, batchSize, truncateEventsToLength, rng, false);
        }


        //Initiate UI-Server
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();
        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));
        uiServer.attach(statsStorage);

        //start training
        System.out.println("Start training...");
        net.setListeners(new ScoreIterationListener(100));
        for (int i = 0; i < nEpochs; i++)
        {
            net.fit(trainIter);
            System.out.println("Epoch: " + i + " complete. Starting evaluation:");

            //Run evaluation
            Evaluation evaluation = net.evaluate(testIter);
            System.out.println(evaluation.stats());
            System.out.println("-----------------------------------------------------------------------------");

            evaluation.getConfusionMatrix();
            System.out.println(evaluation.confusionToString());
        }
    }
}
