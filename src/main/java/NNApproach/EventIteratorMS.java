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
 * Textclassification for machen/schauen with RNN
 *
 * @author Alex Black
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment/SentimentExampleIterator.java
 * @author Thomas Buchegger
 */
public class EventIteratorMS implements DataSetIterator
{
    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private int cursor = 0;
    private final File[] machenFiles;
    private final File[] schauenFiles;

    private final TokenizerFactory tokenizerFactory;

    /**
     * Constructor
     *
     * @param dataDirectory  Path to event datasets
     * @param wordVectors    Pre-trained wordvectors
     * @param batchSize      Size of each minibatch for training
     * @param truncateLength If event exceed
     * @param train          If true: return the training data. If false: return the testing data.
     */
    public EventIteratorMS(String dataDirectory, WordVectors wordVectors, int batchSize, int truncateLength, boolean train) throws IOException
    {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;


        File s = new File(FilenameUtils.concat(dataDirectory, (train ? "train" : "test") + "/1/") + "/");
        File m = new File(FilenameUtils.concat(dataDirectory, (train ? "train" : "test") + "/2/") + "/");

        machenFiles = m.listFiles();
        schauenFiles = s.listFiles();

        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    @Override
    public DataSet next(int num)
    {
        if (cursor >= machenFiles.length + schauenFiles.length) throw new NoSuchElementException();
        try
        {
            return nextDataSet(num);
        } catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException
    {
        //First: load events to String. Alternate machen and schauen events
        List<String> events = new ArrayList<>(num);
        boolean[] machen = new boolean[num];
        for (int i = 0; i < num && cursor < totalExamples(); i++)
        {
            if (cursor % 2 == 0)
            {
                //Load machen review
                int machenNumber = cursor / 2;
                String review = FileUtils.readFileToString(machenFiles[machenNumber]);
                events.add(review);
                machen[i] = true;
            }
            else
            {
                //Load negative review
                int schauenNumber = cursor / 2;
                String review = FileUtils.readFileToString(schauenFiles[schauenNumber]);
                events.add(review);
                machen[i] = false;
            }
            cursor++;
        }

        //Second: tokenize events and filter out unknown words
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
        //Here: we have events.size() examples of varying lengths
        INDArray features = Nd4j.create(new int[]{events.size(), vectorSize, maxLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{events.size(), 2, maxLength}, 'f');    //six labels: sport, politik, kultur, musik, essen, freizeit

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

            int idx = (machen[i] ? 0 : 1);
            int lastIdx = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);   //Set label: [0,1] for negative, [1,0] for machen
            labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);
        }

        return new DataSet(features, labels, featuresMask, labelsMask);
    }

    @Override
    public int totalExamples()
    {
        return machenFiles.length + schauenFiles.length;
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
        return Arrays.asList("machen", "schauen");
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
