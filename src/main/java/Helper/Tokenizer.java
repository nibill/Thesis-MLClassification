package Helper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tokenize event text
 *
 * @author Thomas Buchegger
 */
public class Tokenizer
{
    /**
     * Tokenize a single string
     *
     * @param text
     * @return
     */
    public static ArrayList<String> Tokenize(String text)
    {
        ArrayList<String> words = new ArrayList<>();
        Pattern pattern = Pattern.compile("\\w+", Pattern.UNICODE_CHARACTER_CLASS);
        Matcher m = pattern.matcher(text);
        while (m.find())
        {
            words.add(m.group());
        }

        Collections.sort(words);

        return words;
    }

    /**
     * Tokenize a list of strings
     *
     * @param eventList
     * @return
     */
    public static ArrayList<String> TokenizeList(ArrayList<String> eventList)
    {
        ArrayList<String> events = new ArrayList<>();
        ArrayList<String> tokenizedEvents = new ArrayList<>();
        Pattern pattern = Pattern.compile("\\w+", Pattern.UNICODE_CHARACTER_CLASS);

        // firstly iterate through events
        for (String event : eventList)
        {
            Matcher m = pattern.matcher(event);
            while (m.find())
            {
                events.add(m.group());
            }

            String result = String.join(" ", events);
            tokenizedEvents.add(result);

            events.clear();
        }

        return tokenizedEvents;
    }
}
