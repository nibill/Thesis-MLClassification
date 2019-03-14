package NNApproach;


import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Textclassification with RNN
 *
 * @author Alex Black
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment/SentimentExampleIterator.java
 * @author Thomas Buchegger
 */
public class EventIterator implements DataSetIterator
{
    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private int cursor = 0;
    private final File[] sportFiles;
    private final File[] politikFiles;
    private final File[] kulturFiles;
    private final File[] musikFiles;
    private final File[] essenFiles;
    private final File[] freizeitFiles;
    private final TokenizerFactory tokenizerFactory;

    /**
     * Constructor
     * @param dataDirectory  Path to event datasets
     * @param wordVectors    Pre-trained wordvectors
     * @param batchSize      Size of each minibatch for training
     * @param truncateLength If event exceed
     * @param train          If true: return the training data. If false: return the testing data.
     */
    public EventIterator(String dataDirectory, WordVectors wordVectors, int batchSize, int truncateLength, boolean train) throws IOException
    {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;


        File s = new File(FilenameUtils.concat(dataDirectory, (train ? "train" : "test") + "/1/") + "/");
        File p = new File(FilenameUtils.concat(dataDirectory, (train ? "train" : "test") + "/2/") + "/");
        File k = new File(FilenameUtils.concat(dataDirectory, (train ? "train" : "test") + "/3/") + "/");
        File m = new File(FilenameUtils.concat(dataDirectory, (train ? "train" : "test") + "/4/") + "/");
        File e = new File(FilenameUtils.concat(dataDirectory, (train ? "train" : "test") + "/5/") + "/");
        File f = new File(FilenameUtils.concat(dataDirectory, (train ? "train" : "test") + "/6/") + "/");

        sportFiles = s.listFiles();
        politikFiles = p.listFiles();
        kulturFiles = k.listFiles();
        musikFiles = m.listFiles();
        essenFiles = e.listFiles();
        freizeitFiles = f.listFiles();

        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    @Override
    public DataSet next(int num)
    {
        if (cursor >= sportFiles.length + politikFiles.length + kulturFiles.length + musikFiles.length
            + essenFiles.length + freizeitFiles.length) throw new NoSuchElementException();
        try
        {
            return nextDataSet(num);
        } catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    /**
     * Iterate through datasets
     * @param num
     * @return
     * @throws IOException
     */

    private DataSet nextDataSet(int num) throws IOException
    {
        //First: load events to String. Alternate through all event categories
        List<String> events = new ArrayList<>(num);
        List<Integer> cat = new ArrayList<>();
        for (int i = 0; i < num && cursor < totalExamples(); i++)
        {
            if (cursor % 6 == 1)
            {
                //Load category sport
                int sportEventNumber = cursor / 6;
                String event = FileUtils.readFileToString(sportFiles[sportEventNumber]);
                events.add(event);
                cat.add(i, 1);
            } else if (cursor % 6 == 2)
            {
                //Load category politik
                int politikEventNumber = cursor / 6;
                String event = FileUtils.readFileToString(politikFiles[politikEventNumber]);
                events.add(event);
                cat.add(i, 2);
            } else if (cursor % 6 == 3)
            {
                //Load category kultur
                int kulturEventNumber = cursor / 6;
                String event = FileUtils.readFileToString(kulturFiles[kulturEventNumber]);
                events.add(event);
                cat.add(i, 3);
            } else if (cursor % 6 == 4)
            {
                //Load category musik
                int musikEventNumber = cursor / 6;
                String event = FileUtils.readFileToString(musikFiles[musikEventNumber]);
                events.add(event);
                cat.add(i, 4);
            } else if (cursor % 6 == 5)
            {
                //Load category essen
                int essenEventNumber = cursor / 6;
                String event = FileUtils.readFileToString(essenFiles[essenEventNumber]);
                events.add(event);
                cat.add(i, 5);
            } else if (cursor % 6 == 0)
            {
                //Load category freizeit
                int freizeitEventNumber = cursor / 6;
                String event = FileUtils.readFileToString(freizeitFiles[freizeitEventNumber]);
                events.add(event);
                cat.add(i, 6);
            }
            cursor++;
        }

        //Second: tokenize reviews and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(events.size());
        int maxLength = 0;
        for (String s : events)
        {
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for (String t : tokens)
            {
                if (wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength, tokensFiltered.size());
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if (maxLength > truncateLength) maxLength = truncateLength;

        //Create data for training
        INDArray features = Nd4j.create(new int[]{events.size(), vectorSize, maxLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{events.size(), 6, maxLength}, 'f');    //six labels: sport, politik, kultur, musik, essen, freizeit

        //Because we are dealing with events of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(events.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(events.size(), maxLength);

        for (int i = 0; i < events.size(); i++)
        {
            List<String> tokens = allTokens.get(i);

            // Get the truncated sequence length of document (i)
            int seqLength = Math.min(tokens.size(), maxLength);

            // Get all wordvectors for the current document and transpose them to fit the 2nd and 3rd feature shape
            final INDArray vectors = wordVectors.getWordVectors(tokens.subList(0, seqLength)).transpose();

            // Put wordvectors into features array at the following indices:
            // 1) Document (i)
            // 2) All vector elements which is equal to NDArrayIndex.interval(0, vectorSize)
            // 3) All elements between 0 and the length of the current sequence
            features.put(new INDArrayIndex[]
                {
                    NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, seqLength)
                }, vectors);

            // Assign "1" to each position where a feature is present, that is, in the interval of [0, seqLength)
            featuresMask.get(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength)}).assign(1);

            int idx = 0;
            if (cat.get(i) % 6 == 1)
                idx = 0;
            else if (cat.get(i) % 6 == 2)
                idx = 1;
            else if (cat.get(i) % 6 == 3)
                idx = 2;
            else if (cat.get(i) % 6 == 4)
                idx = 3;
            else if (cat.get(i) % 6 == 5)
                idx = 4;
            else if (cat.get(i) % 6 == 0)
                idx = 5;

            int lastIdx = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);
            labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features, labels, featuresMask, labelsMask);
    }

    @Override
    public int totalExamples()
    {
        return sportFiles.length + politikFiles.length + kulturFiles.length + musikFiles.length + essenFiles.length + freizeitFiles.length;
    }

    @Override
    public int inputColumns()
    {
        return vectorSize;
    }

    @Override
    public int totalOutcomes()
    {
        return 6;
    }

    @Override
    public void reset()
    {
        cursor = 0;
    }

    public boolean resetSupported()
    {
        return true;
    }

    @Override
    public boolean asyncSupported()
    {
        return true;
    }

    @Override
    public int batch()
    {
        return batchSize;
    }

    @Override
    public int cursor()
    {
        return cursor;
    }

    @Override
    public int numExamples()
    {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor)
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels()
    {
        return Arrays.asList("sport", "politik", "kultur", "musik", "essen", "freizeit");
    }

    @Override
    public boolean hasNext()
    {
        return cursor < numExamples();
    }

    @Override
    public DataSet next()
    {
        return next(batchSize);
    }

    @Override
    public void remove()
    {

    }

    @Override
    public DataSetPreProcessor getPreProcessor()
    {
        throw new UnsupportedOperationException("Not implemented");
    }
}
