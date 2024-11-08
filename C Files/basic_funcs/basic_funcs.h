/**
 * Calculates the number of different types of coins needed to make a given amount of cents.
 * This function uses the denominations of quarters, dimes, nickels, and pennies, starting 
 * from the largest denomination to the smallest to minimize the total number of coins used.
 */
unsigned char coin_types(unsigned int cents);

/**
 * Calculates the number of weeks needed to cover a specified total lecture time 
 * given the maximum lecture duration and the maximum number of lectures per week.
 * This function rounds up to ensure all required lecture time is accommodated.
 */
unsigned short weeks_in_term(unsigned short total_mins,
                             unsigned short max_mins_per_lecture,
                             unsigned char max_lectures_week);

/**
 * Returns the last letter of the English name of the month corresponding to the given number.
 * Valid input is between 1 (January) and 12 (December). If the input is out of this range,
 * a space character is returned.
 */                             
char month_ends_in(unsigned char month);

/**
 * Returns the number of segments lit for a single digit (0-9) on a seven-segment display.
 * The segment counts are predefined for each digit.
 */
unsigned int segments_for_digit(int digit);

/**
 * Recursively calculates the total number of segments lit to display a given number 
 * on a seven-segment display. It extracts each digit of the number arithmetically.
 */
unsigned int lit_segments(unsigned int number);