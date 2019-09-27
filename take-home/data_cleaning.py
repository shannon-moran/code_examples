'''
Appendix A2: data_processing.py
For production code, we'd want to document in/output types, add unit tests, etc.
Not going to worry about that here.
'''

import pandas as pd
import numpy as np

def clean_contacts(contacts):
    # convert these columns to datetime
    datetime_columns = ['ts_interaction_first',
                  'ts_reply_at_first',
                  'ts_accepted_at_first',
                  'ts_booking_at',
                  'ds_checkin_first',
                  'ds_checkout_first']
    for col in datetime_columns:
        contacts[col] = pd.to_datetime(contacts[col])

    # converts times into fractional days
    seconds_day = (24*60*60)
    contacts['ts_first_response'] = (contacts['ts_reply_at_first'] - contacts['ts_interaction_first']).dt.total_seconds()/seconds_day
    contacts['ts_acceptance'] = (contacts['ts_accepted_at_first'] - contacts['ts_interaction_first']).dt.total_seconds()/seconds_day
    contacts['ts_booking'] = (contacts['ts_booking_at'] - contacts['ts_interaction_first']).dt.total_seconds()/seconds_day
    contacts['ds_stay'] = (contacts['ds_checkout_first'] - contacts['ds_checkin_first']).dt.total_seconds()/seconds_day
    contacts['ds_until_booking'] = np.ceil((contacts['ds_checkin_first'] - contacts['ts_interaction_first']).dt.total_seconds()/seconds_day)
    # Need to up to account for same-day bookings

    # data cleanliness
    contacts = contacts[contacts.ts_first_response.isnull()].append(contacts[contacts.ts_first_response>=0])
    contacts = contacts[contacts.guest_user_stage_first!='-unknown-']
    contacts = contacts[contacts.ds_until_booking<365]
    contacts = contacts[contacts.ds_until_booking>=-1] # there are a few same-day bookings that were made after midnight
    contacts.loc[contacts.ds_until_booking==-1,'ds_until_booking']=0 # correct for thsoe same-day bookings
    return contacts

def clean_listings(listings):
    listings = listings[listings['total_reviews']>=0]
    return listings

def clean_users(users):
    users = users.drop_duplicates(subset='id_user_anon', keep='first')
    return users

def clean_all_data(contacts,listings,users):
    contacts = clean_contacts(contacts)
    listings = clean_listings(listings)
    users = clean_users(users)
    return contacts, listings, users

def merge_data(contacts, listings, users):
    contacts, listings, users = clean_all_data(contacts,listings,users)

    # add in listing data
    data = pd.merge(contacts,listings,how='left',on='id_listing_anon')

    # add in user data
    users_guests = users.rename(columns={'id_user_anon':'id_guest_anon',
                                         'country':'guest_country',
                                         'words_in_user_profile':'wds_in_guest_profile'})

    users_hosts = users.rename(columns={'id_user_anon':'id_host_anon',
                                        'country':'host_country',
                                        'words_in_user_profile':'wds_in_host_profile'})

    data = pd.merge(data,users_guests,how='left',on='id_guest_anon')
    data = pd.merge(data,users_hosts,how='left',on='id_host_anon')
    return data
