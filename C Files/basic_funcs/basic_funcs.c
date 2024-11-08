#include <stdlib.h>
#include <stdio.h>
#include "hw1.h"

unsigned char coin_types(unsigned int cents){
    int coins[] = {25, 10, 5, 1};
    int types_of_coins = 0;

    for (int i = 0; i < 4; i++) {
        if (cents >= coins[i]) {
            types_of_coins++;  
            cents -= (cents / coins[i]) * coins[i];  
        }
    }
    return types_of_coins;
}

unsigned short weeks_in_term(unsigned short total_mins, unsigned short max_mins_per_lecture, unsigned char max_lectures_week){
    int total_lectures = (total_mins + max_mins_per_lecture - 1) / max_mins_per_lecture;  // Round up

    int total_weeks = (total_lectures + max_lectures_week - 1) / max_lectures_week;  // Round up

    return total_weeks;
}

char month_ends_in(unsigned char month){
    switch (month) {
        case 1: return 'y'; 
        case 2: return 'y'; 
        case 3: return 'h'; 
        case 4: return 'l'; 
        case 5: return 'y'; 
        case 6: return 'e'; 
        case 7: return 'y'; 
        case 8: return 't'; 
        case 9: return 'r'; 
        case 10: return 'r'; 
        case 11: return 'r'; 
        case 12: return 'r'; 
        default: return ' '; 
    }
}

unsigned int segments_for_digit(int digit) {
    int segments[] = {6, 2, 5, 5, 4, 5, 6, 3, 7, 6};
    return segments[digit];
}

unsigned int lit_segments(unsigned int number){
    if (number == 0) {
        return segments_for_digit(0);
    } else if (number < 10) {
        return segments_for_digit(number);
    } else {
        return segments_for_digit(number % 10) + lit_segments(number / 10);
    }
}