package Helper;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Preprocess events
 *
 * @author Thomas Buchegger
 */
public class TextPreprocessing
{
    private final static String weightingWords = "Resources" + File.separator + "CategorizedWords" + File.separator;

    /**
     * Replace Umlaute
     * methode von https://github.com/danielnaber/openthesaurus/blob/master/src/java/com/vionto/vithesaurus/tools/StringTools.java
     *
     * @param eventList List with events
     * @return Cleaned events
     */
    public static ArrayList<String> CleanWord(ArrayList<String> eventList)
    {
        ArrayList<String> cleanedEvents = new ArrayList<>();

        for (String events : eventList)
        {
            events = events.replaceAll("[.!?,]", "")
                .replace("Ä", "Ae").replace("ä", "ae")
                .replace("Ü", "Ue").replace("ü", "ue")
                .replace("Ö", "Oe").replace("ö", "oe")
                .replace("ß", "ss").replaceAll("[^A-Za-z]", " ")
                .replace("  ", " ");

            cleanedEvents.add(events);
        }

        return cleanedEvents;
    }

    /**
     * Remove URL from events
     *
     * @param eventList List with events
     * @return Cleaned events without URLs
     */
    private static ArrayList<String> RemoveUrl(ArrayList<String> eventList)
    {
        ArrayList<String> cleanedEvents = new ArrayList<>();
        String urlPattern = "((https?|ftp|gopher|telnet|file|Unsure|http):((//)|(\\\\))+[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*)";
        Pattern p = Pattern.compile(urlPattern, Pattern.CASE_INSENSITIVE);

        for (String event : eventList)
        {
            Matcher m = p.matcher(event);
            int i = 0;
            while (m.find())
            {
                try
                {
                    // if URL found, replace with nothing
                    event = event.replaceAll(m.group(i), "").trim();
                } catch (PatternSyntaxException ex)
                {
                    System.out.println(ex);
                } catch (IllegalArgumentException ex)
                {
                    System.out.println(ex);
                } catch (IndexOutOfBoundsException ex)
                {
                    System.out.println(ex);
                }
            }
            cleanedEvents.add(event);
        }

        return cleanedEvents;
    }

    /**
     * Sorts words in events alphabetically
     *
     * @param unsortedList List with events
     * @param eventList    Sort list or words in events
     * @return Sorted events
     */
    private static ArrayList<String> SortList(ArrayList<String> unsortedList, boolean eventList)
    {
        ArrayList<String> sortedList = new ArrayList<>();

        if (eventList)
        {
            for (String event : unsortedList)
            {
                ArrayList<String> words = new ArrayList<String>(Arrays.asList(event.split(" ")));

                Collections.sort(words);

                sortedList.add(String.join(" ", words));
            }
        } else
        {
            Collections.sort(unsortedList);

            sortedList = unsortedList;
        }
        return sortedList;
    }

    /**
     * Weights words based on their appearance
     *
     * @param eventList List with events
     * @param cat       Which category should be loaded
     * @param weighting Weight intensity
     * @return List with weighted words
     */
    private static ArrayList<String> DoWeighting(ArrayList<String> eventList, int cat, int weighting)
    {
        // load list with word to be weighted
        ArrayList<String> wheightingList = FileHandler.GetWeightingWordList(weightingWords, cat);
        wheightingList = Tokenizer.Tokenize(String.join(" ", wheightingList));

        // sort list for faster lookup
        ArrayList<String> sortedWeightingList = SortList(wheightingList, false);

        ArrayList<String> weightedEvent = new ArrayList<>();

        // iterate through events and check whether a certain word in an event appears in the weighting list
        for (String event : eventList)
        {
            ArrayList<String> words = new ArrayList<>(Arrays.asList(event.split(" ")));

            HashSet<String> hsEvent = new HashSet<>(words);

            for (String word : sortedWeightingList)
            {
                if (hsEvent.contains(word))
                {
                    // add the word as many times as the weighting indicates
                    for (int i = 1; i <= weighting - 1; i++)
                    {
                        words.add(word);
                    }
                }
            }

            // add all words
            weightedEvent.add(String.join(" ", words));
        }
        return weightedEvent;
    }

    /**
     * Which preprocessing should be done
     *
     * @param unpreprocssedList List with events
     * @return List with preprocessed events
     */
    public static ArrayList<String> DoPreprocessing(ArrayList<String> unpreprocssedList)
    {
        ArrayList<String> preprocessedList;

        //preprocessedList = RemoveUrl(unpreprocssedList);
        preprocessedList = Tokenizer.TokenizeList(unpreprocssedList);
        preprocessedList = CleanWord(preprocessedList);
        preprocessedList = Stopwords.RemoveStopwords(preprocessedList);
        //preprocessedList = SortList(preprocessedList, true);

        return preprocessedList;
    }

    /**
     * Saves the preprocessed events to files
     *
     * @param eventList  List with events
     * @param cat        Which category do the events belong to
     * @param outputPath Path to where files should be saved
     */
    public static void CopyFiles(ArrayList<String> eventList, int cat, String outputPath)
    {
        ArrayList<String> finalizedEventText;

        finalizedEventText = DoPreprocessing(eventList);

        // weighting list currently too short to make a difference, so not used
        //finalizedEventText = DoWeighting(finalizedEventText, cat, 2);

        int i = 1;
        for (String event : finalizedEventText)
        {
            try
            {
                String outPath = outputPath + File.separator + cat + File.separator + "ct" + i + ".txt";

                // clear directories
                FileHandler.DeleteFiles(outPath);

                File file = new File(outPath);
                FileWriter fileWriter = new FileWriter(file);
                fileWriter.write(event);
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e)
            {
                e.printStackTrace();
            }

            i++;
        }
    }
}
