from src.plotting import plot_nr_credict_applications


def make_plots(df_appl_tv):
    # plot all companies:
    total_nr_applications = df_appl_tv.groupby(['yearmonth_dt']).nr_credit_applications.sum()
    plot_nr_credict_applications(total_nr_applications, 'All companies')

    # appl_per_client = df_appl_tv.groupby(['client_nr']).nr_credit_applications.sum()
    # appl_per_client[appl_per_client > 0]

    # plot company one:
    company_three = df_appl_tv[df_appl_tv.client_nr == 3]
    company_three_nr_applications = company_three.groupby(['yearmonth_dt']).nr_credit_applications.sum()
    plot_nr_credict_applications(company_three_nr_applications, 'Client #3')
