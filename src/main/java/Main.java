import Helper.*;
import NNApproach.CNNClassification;
import NNApproach.RNNClassification;
import NNApproach.Vectorizer;
import VectorApproach.NaiveBayesSTWV;

import javax.xml.bind.annotation.XmlType;
import java.io.BufferedReader;
import java.io.Console;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class Main
{
    private final static String ARFF_INPUT_FILE = "Data" + File.separator + "Input";

    private final static String WHOLE_TEXTS_PATH = "Data" + File.separator + "Input" + File.separator + "WholeText" + File.separator;
    private final static String PREPROCESSED_CAT_PATH = "Data" + File.separator + "Input" + File.separator + "PreprocessedCat" + File.separator;

    private final static String WHOLE_TEXTS_MS_PATH = "Data" + File.separator + "Input" + File.separator + "WholeTextMS" + File.separator;
    private final static String PREPROCESSED_MS_PATH = "Data" + File.separator + "Input" + File.separator + "PreprocessedMS" + File.separator;

    private final static String TRAIN_TEST_PATH = "Data" + File.separator + "Input" + File.separator + "TrainTest" + File.separator;
    private final static String TRAIN_TEST_PATHMS = "Data" + File.separator + "Input" + File.separator + "TrainTestMS" + File.separator;

    private final static String WIKI_VECTORS_PATH = "Resources" + File.separator + "WordVectors" + File.separator + "wiki.de.vec";

    public static void main(String[] args) throws Exception
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        boolean cont = true;

        String format = "%-5s%s%n";
        System.out.println("-----------------------------------------------------------------------------");

        System.out.println("Would you like to preprocess the input data?");
        System.out.printf(format, "Yes, category           - ", 1);
        System.out.printf(format, "Yes, schauen/machen     - ", 2);
        System.out.printf(format, "Use unpreprocessed data - ", 3);
        System.out.printf(format, "Continue                - ", 4);

        String input = br.readLine();
        String ms;

        if (input.equals("1"))
        {
            // load all category events
            ArrayList<String> eventsCat1 = FileHandler.GetEventsCat1(WHOLE_TEXTS_PATH + "1");
            ArrayList<String> eventsCat2 = FileHandler.GetEventsCat2(WHOLE_TEXTS_PATH + "2");
            ArrayList<String> eventsCat3 = FileHandler.GetEventsCat3(WHOLE_TEXTS_PATH + "3");
            ArrayList<String> eventsCat4 = FileHandler.GetEventsCat4(WHOLE_TEXTS_PATH + "4");
            ArrayList<String> eventsCat5 = FileHandler.GetEventsCat5(WHOLE_TEXTS_PATH + "5");
            ArrayList<String> eventsCat6 = FileHandler.GetEventsCat6(WHOLE_TEXTS_PATH + "6");

            ArrayList<ArrayList<String>> eventLists = new ArrayList<>();
            eventLists.add(eventsCat1);
            eventLists.add(eventsCat2);
            eventLists.add(eventsCat3);
            eventLists.add(eventsCat4);
            eventLists.add(eventsCat5);
            eventLists.add(eventsCat6);

            for (int i = 0; i < eventLists.size(); i++)
            {
                TextPreprocessing.CopyFiles(eventLists.get(i), i + 1, PREPROCESSED_CAT_PATH);
            }
            System.out.println("Postprocessing done!");
        }
        else if (input.equals("2"))
        {
            // load all category events
            ArrayList<String> eventsMachen = FileHandler.GetEventsMachen(WHOLE_TEXTS_MS_PATH + "1");
            ArrayList<String> eventsSchauen = FileHandler.GetEventsSchauen(WHOLE_TEXTS_MS_PATH + "2");

            ArrayList<ArrayList<String>> eventLists = new ArrayList<>();
            eventLists.add(eventsSchauen);
            eventLists.add(eventsMachen);

            for (int i = 0; i < eventLists.size(); i++)
            {
                TextPreprocessing.CopyFiles(eventLists.get(i), i + 1, PREPROCESSED_MS_PATH);
            }
            System.out.println("Postprocessing done!");
        }


        while (cont)
        {
            System.out.println("-----------------------------------------------------------------------------");
            System.out.println("Would you like to classify into categories or into machen/schauen?");
            System.out.printf(format, "Categories     - ", 1);
            System.out.printf(format, "machen/schauen - ", 2);
            System.out.println("-----------------------------------------------------------------------------");

            ms = br.readLine();

            String path;

            if (input.equals("3"))
            {
                if (ms.equals("1"))
                {

                    path = WHOLE_TEXTS_PATH;
                }
                else
                {
                    path = WHOLE_TEXTS_MS_PATH;
                }
            }
            else
            {
                if (ms.equals("1"))
                {

                    path = PREPROCESSED_CAT_PATH;
                }
                else
                {
                    path = PREPROCESSED_MS_PATH;
                }
            }


            System.out.println("-----------------------------------------------------------------------------");
            System.out.println("Please choose your classifier:");
            System.out.printf(format, "Naive Bayes   - ", 1);
            System.out.printf(format, "CNN           - ", 2);
            System.out.printf(format, "RNN           - ", 3);
            System.out.println("-----------------------------------------------------------------------------");

            input = br.readLine();

            if (input.equals("1"))
            {
                boolean tfidf = false;
                System.out.println("With TF-IDF?");
                System.out.printf(format, "Yes   - ", 1);
                System.out.printf(format, "No    - ", 2);
                input = br.readLine();

                if (input.equals("1"))
                {
                    tfidf = true;
                }
                else
                {
                    tfidf = false;
                }

                NaiveBayesSTWV.DoNBSTWVEval(path, ARFF_INPUT_FILE, UsedClassifier.MNNB, tfidf);
            }
            else if (input.equals("2"))
            {
                if (ms.equals("1"))
                {
                    CNNClassification.CNNClassify(WIKI_VECTORS_PATH, TRAIN_TEST_PATH, true);
                }
                else
                {
                    CNNClassification.CNNClassify(WIKI_VECTORS_PATH, TRAIN_TEST_PATHMS, false);
                }

            }
            else if (input.equals("3"))
            {
                RNNClassification.RNNClassify(WIKI_VECTORS_PATH, TRAIN_TEST_PATHMS);
            }
            else
            {
                System.out.println("Invalid number!");
            }

            System.out.println("-----------------------------------------------------------------------------");
            System.out.println("Would you like to try again?");
            System.out.printf(format, "Yes   - ", 1);
            System.out.printf(format, "No    - ", 2);

            input = br.readLine();
            if (input.equals("1"))
            {
                cont = true;
            }
            else
            {
                cont = false;
            }
        }
    }
}
