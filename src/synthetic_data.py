"""
Calibrated Synthetic Mobile Money Data Generator v3.0

Location: src/synthetic_data.py

This generator creates realistic per-user transaction datasets calibrated against
actual mobile money patterns from Ghana (593 real transactions analyzed from Table 5).

Key Features:
- Generates 10,000 individual user transaction files
- Includes CREDIT (loan disbursement) and LOAN_REPAYMENT transactions
- Models credit risk behavior with multiple user archetypes
- Calibrated to real Ghanaian mobile money patterns
- Granular DEBIT provider model with provider-specific amounts
- Realistic incoming transactions (~28% of transfers)
- Accurate fee/e-levy models matching real data
- 16-column output matching real transaction export format

Loan System (based on MTN QwikLoan Ghana):
- Loan amounts: GHS 25 - GHS 1,000
- Interest rate: 6.9% (30-day term)
- Penalty rate: 12.5% (on default)
- Providers: QWIKLOAN, XPRESSLOAN, XTRACASH, FIDO, CEDISPAY

Credit Risk Labels:
- 0: Good (repaid in full within term)
- 1: Late (repaid but after 30 days)
- 2: Default (failed to repay within 60 days)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import string
from pathlib import Path


class CalibratedMoMoDataGenerator:
    """
    Generate realistic mobile money transaction data with credit/loan functionality.
    Outputs individual CSV files per user.
    """

    # Loan providers in Ghana
    LOAN_PROVIDERS = ['QWIKLOAN', 'XPRESSLOAN', 'XTRACASH', 'FIDO', 'CEDISPAY']

    # Ghanaian first names (from real transaction data)
    GHANAIAN_FIRST_NAMES = [
        'Ibrahim', 'Beatrice', 'Jennifer', 'Cynthia', 'Edmund', 'Portia',
        'Irene', 'Helena', 'Fatimatu', 'Ruth', 'Edna', 'Doris', 'Kate',
        'Margaret', 'Hannah', 'Vivian', 'Comfort', 'Adizatu', 'Theresa',
        'Rosemond', 'Belinda', 'Grace', 'Mary', 'Patience', 'Vincent',
        'Samuel', 'Prince', 'Michael', 'Isaac', 'Bernard', 'Emmanuel',
        'Daniel', 'Paul', 'Elias', 'Titus', 'Kwadwo', 'Prosper',
        'Rodgers', 'Jeffery', 'Amos', 'Joseph', 'Eric', 'Sampson',
        'Derrick', 'Steve', 'Gabriel', 'Ishmael', 'David', 'Frederick',
        'Juliana', 'Ellen', 'Linda', 'Rose', 'Faustina', 'Sarah',
        'Ivy', 'Mavis', 'Abena', 'Gloria', 'Vida', 'Mercy', 'Naomi',
        'Susanna', 'Christiana', 'Getrude', 'Esther', 'Moses', 'Newman',
        'Nathaniel', 'Standford', 'Kwabena', 'Kofi', 'Kingsley', 'Dennis'
    ]

    # Ghanaian last names (from real transaction data)
    GHANAIAN_LAST_NAMES = [
        'Ahmad Nazir', 'Akanjam', 'Nyarko', 'Donkor', 'Annan', 'Marfo',
        'Agyemang', 'Agyei', 'Peprah', 'Ofori', 'Osei', 'Asantewaa',
        'Kwarteng', 'Dankwah', 'Kyei', 'Attabra', 'Ashiagbor', 'Agbenya',
        'Yeboah', 'Acheampong', 'Aboagye', 'Fuseini', 'Acquah', 'Arhin',
        'Amankwaa', 'Dzehor', 'Adu', 'Appiah', 'Sarfo', 'Tetteh',
        'Ampong', 'Ansah', 'Boakye', 'Owusu', 'Mensah', 'Laryea',
        'Adumuah', 'Konadu', 'Nsiah', 'Fynn', 'Danquah', 'Sarpong',
        'Opoku', 'Armah', 'Afoakwa', 'Vikpe', 'Gyamfi', 'Larbi',
        'Abotare', 'Ntibiah', 'Todzi', 'Anaba', 'Hanson', 'Mill'
    ]

    # DEBIT provider model (from Table 5 analysis)
    DEBIT_PROVIDERS = {
        'one4all.sp': {
            'weight': 0.48,
            'to_name': 'one4all.sp',
            'to_acct': '67076778',
            'to_no': '0',
            'amount_type': 'discrete',
            'amounts': [3, 3.5, 5, 7, 7.5, 9, 10, 11.5, 15],
            'amount_weights': [0.05, 0.02, 0.15, 0.02, 0.02, 0.02, 0.55, 0.02, 0.15],
        },
        'cis': {
            'weight': 0.18,
            'to_name': 'cis',
            'to_acct': '53169694',
            'to_no': '0',
            'amount_type': 'fixed',
            'fixed_amount': 5,
        },
        'HubtelPOS.sp': {
            'weight': 0.12,
            'to_name': 'POS.Inv',
            'to_acct': '61973894',
            'to_no': '233547169606',
            'amount_type': 'fixed',
            'fixed_amount': 10,
        },
        'cisnew': {
            'weight': 0.10,
            'to_name': 'cisnew',
            'to_acct': '62525623',
            'to_no': '0',
            'amount_type': 'fixed',
            'fixed_amount': 3,
        },
        'J4U': {
            'weight': 0.04,
            'to_name': 'J4U',
            'to_acct': '72767502',
            'to_no': '0',
            'amount_type': 'fixed',
            'fixed_amount': 12,
        },
        'Cell.sp': {
            'weight': 0.04,
            'to_name': 'Cellulant',
            'to_acct': '68688336',
            'to_no': '233554680944',
            'amount_type': 'range',
            'amount_min': 100,
            'amount_max': 320,
        },
        'calpush.sp': {
            'weight': 0.03,
            'to_name': 'calpush.sp',
            'to_acct': '46831697',
            'to_no': '0',
            'amount_type': 'range',
            'amount_min': 120,
            'amount_max': 360,
        },
        'pstack.sp': {
            'weight': 0.02,
            'to_name': 'Paystack Ghana Limited',
            'to_acct': '71167196',
            'to_no': '233257837233',
            'amount_type': 'range',
            'amount_min': 20,
            'amount_max': 110,
        },
    }

    # PAYMENT sub-types (from Table 5 analysis)
    PAYMENT_SUBTYPES = {
        'airtime': {
            'weight': 0.75,
            'to_name': 'MTNONLINEAIRTIMEVENDOR',
            'to_acct': '39011161',
            'to_no': '0',
            'ova': 'MTNONLINEAIRTIMEVENDO',
            'amounts': [1, 2.5, 5, 5.5, 6, 10, 10.5, 12, 15, 15.5, 25],
            'amount_weights': [0.04, 0.04, 0.14, 0.10, 0.12, 0.14, 0.04, 0.10, 0.14, 0.06, 0.08],
        },
        'cross_network': {
            'weight': 0.25,
            'to_name_options': ['VODAFONE PUSH OVA', 'TIGO PUSH OVA'],
            'to_acct_options': ['54814522', '71396410'],
            'ova_options': ['mmipush', 'mmipushtigo'],
            'amount_min': 15,
            'amount_max': 120,
        },
    }

    # Incoming transaction source categories
    INCOMING_FAMILY_AMOUNTS = [51, 101, 102, 122, 130, 152, 200, 202, 210, 303, 355, 360, 505, 520]
    INCOMING_FAMILY_REFS = ['from mum', '1234', 'mum', 'from mu', 'Easter gift', 'Bottle water']

    INCOMING_FRIEND_REFS = [
        '1', 'Chale', 'Food', '0000', 'W', 'Adombi', 'Part payment',
        'PiRSquared', 'Gobe and calabash', 'Bills', 'Someonesbill',
        '123', 'Oa', 'Danny', 'Atm', 'Gh', '347', 'hard drive',
    ]

    # Credit behavior archetypes
    CREDIT_ARCHETYPES = {
        'non_borrower': {
            'weight': 0.40,
            'takes_loans': False,
            'description': 'Never takes loans'
        },
        'responsible_borrower': {
            'weight': 0.35,
            'takes_loans': True,
            'loan_frequency': 'regular',
            'repayment_behavior': 'on_time',
            'credit_limit_growth': True,
            'default_probability': 0.0,
            'description': 'Takes loans, always repays on time'
        },
        'occasional_borrower': {
            'weight': 0.15,
            'takes_loans': True,
            'loan_frequency': 'occasional',
            'repayment_behavior': 'variable',
            'credit_limit_growth': True,
            'default_probability': 0.02,
            'description': 'Infrequent loans, variable repayment timing'
        },
        'risky_borrower': {
            'weight': 0.08,
            'takes_loans': True,
            'loan_frequency': 'frequent',
            'repayment_behavior': 'often_late',
            'credit_limit_growth': False,
            'default_probability': 0.15,
            'description': 'Takes loans near limit, sometimes late'
        },
        'defaulter': {
            'weight': 0.02,
            'takes_loans': True,
            'loan_frequency': 'aggressive',
            'repayment_behavior': 'default',
            'credit_limit_growth': False,
            'default_probability': 1.0,
            'description': 'Takes loans, fails to repay'
        }
    }

    def __init__(self,
                 n_users=10000,
                 avg_transactions_per_user=15,
                 start_date='2024-01-01',
                 duration_days=180,
                 output_dir='data/user_transactions',
                 calibration_file='data/real_data_calibration.json'):
        """
        Initialize generator with calibration parameters.

        Args:
            n_users: Number of unique users to generate
            avg_transactions_per_user: Average transactions per user
            start_date: Start date of observation period
            duration_days: Length of observation period in days
            output_dir: Directory to save individual user CSV files
            calibration_file: Path to JSON file with real data parameters
        """
        self.n_users = n_users
        self.avg_transactions_per_user = avg_transactions_per_user
        self.start_date = pd.to_datetime(start_date)
        self.duration_days = duration_days
        self.end_date = self.start_date + timedelta(days=duration_days)
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load calibration parameters
        with open(calibration_file, 'r') as f:
            self.calibration = json.load(f)

        # Loan parameters (calibrated to Ghana mobile money loans)
        self.loan_params = {
            'min_amount': 25,
            'max_amount': 1000,
            'interest_rate': 0.069,
            'penalty_rate': 0.125,
            'term_days': 30,
            'min_active_days': 90,
            'min_transactions': 15,
            'initial_limit': 50,
            'max_limit': 1000,
            'limit_growth_rate': 1.25
        }

        # Pre-compute DEBIT provider selection arrays
        providers = list(self.DEBIT_PROVIDERS.keys())
        weights = [self.DEBIT_PROVIDERS[p]['weight'] for p in providers]
        self._debit_provider_names = providers
        self._debit_provider_weights = np.array(weights) / sum(weights)

        # Pre-compute PAYMENT subtype selection
        subtypes = list(self.PAYMENT_SUBTYPES.keys())
        pweights = [self.PAYMENT_SUBTYPES[s]['weight'] for s in subtypes]
        self._payment_subtype_names = subtypes
        self._payment_subtype_weights = np.array(pweights) / sum(pweights)

        # Create recipient pool
        self._create_recipient_pool()

        print(f"Initialized generator for {n_users:,} users")
        print(f"Output directory: {self.output_dir}")
        print("Calibration loaded:")
        print(f"  - Amount: mu={self.calibration['amount_lognormal_mu']:.2f}, "
              f"sigma={self.calibration['amount_lognormal_sigma']:.2f}")
        print(f"  - Balance: mean={self.calibration['balance_mean']:.2f}")

    def _generate_name(self):
        """Generate a realistic Ghanaian name."""
        first = np.random.choice(self.GHANAIAN_FIRST_NAMES)
        last = np.random.choice(self.GHANAIAN_LAST_NAMES)
        if np.random.random() < 0.5:
            return f"{first.upper()} {last.upper()}"
        return f"{first} {last}"

    def _create_recipient_pool(self):
        """Create pool of potential transaction recipients with realistic names."""
        self.recipient_pool = [
            {
                'phone': f'233{np.random.randint(240000000, 600000000)}',
                'name': self._generate_name(),
                'account': str(np.random.randint(39000000, 90000000)),
            }
            for _ in range(1000)
        ]

    def _assign_credit_archetype(self):
        """Randomly assign a credit behavior archetype to a user."""
        archetypes = list(self.CREDIT_ARCHETYPES.keys())
        weights = [self.CREDIT_ARCHETYPES[a]['weight'] for a in archetypes]
        return np.random.choice(archetypes, p=weights)

    def _generate_incoming_contacts(self, n):
        """Generate a set of incoming contacts for a user."""
        contacts = []
        for _ in range(n):
            contacts.append({
                'phone': f'233{np.random.randint(240000000, 600000000)}',
                'name': self._generate_name(),
                'account': str(np.random.randint(39000000, 90000000)),
            })
        return contacts

    def generate_user_profile(self, user_id):
        """Generate a single user profile with credit behavior."""

        credit_archetype = self._assign_credit_archetype()
        archetype_config = self.CREDIT_ARCHETYPES[credit_archetype]

        profile = {
            'user_id': f'USER_{user_id:06d}',
            'phone_number': f'233{np.random.randint(200000000, 600000000)}',
            'account_number': str(np.random.randint(39000000, 90000000)),
            'user_name': self._generate_name().upper(),

            # Credit behavior
            'credit_archetype': credit_archetype,
            'takes_loans': archetype_config['takes_loans'],

            # Initial balance
            'initial_balance': max(50, np.random.normal(
                self.calibration['balance_mean'],
                self.calibration['balance_std'] * 0.5
            )),

            # Transaction amount parameters
            'amount_mu': np.random.normal(
                self.calibration['amount_lognormal_mu'], 0.2
            ),
            'amount_sigma': max(0.5, np.random.normal(
                self.calibration['amount_lognormal_sigma'], 0.1
            )),

            # Transaction frequency (hours between transactions)
            'hours_between_txns': max(2, np.random.gamma(
                shape=2,
                scale=abs(self.calibration['transaction_frequency_hours']) / 2
            )),

            # Transaction type preferences (calibrated to Table 5)
            'pref_transfer': max(0.3, min(0.8, np.random.normal(0.612, 0.08))),
            'pref_debit': max(0.1, min(0.35, np.random.normal(0.221, 0.04))),
            'pref_payment_send': max(0.03, min(0.15, np.random.normal(0.088, 0.03))),
            'pref_payment': max(0.01, min(0.08, np.random.normal(0.034, 0.015))),
            'pref_cashout': max(0.02, min(0.10, np.random.normal(0.040, 0.015))),
            'pref_adjustment': max(0.001, min(0.015, np.random.normal(0.005, 0.002))),

            # Temporal preferences
            'pref_weekend': np.random.beta(2, 5),
            'pref_night': np.random.beta(1, 10),
            'pref_hour': np.random.choice([9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]),

            # Balance management
            'min_balance_threshold': np.random.uniform(10, 100),

            # Social network
            'typical_recipients': int(np.random.gamma(shape=3, scale=10)),

            # Account start date (staggered)
            'account_start_day': np.random.randint(0, 14),

            # Incoming transaction contacts
            'incoming_contacts': self._generate_incoming_contacts(np.random.randint(3, 8)),
            'has_family_support': np.random.random() < 0.5,
            'has_bank_link': np.random.random() < 0.3,
        }

        # Add loan-specific parameters if user takes loans
        if profile['takes_loans']:
            profile.update(self._generate_loan_profile(archetype_config))

        return profile

    def _generate_loan_profile(self, archetype_config):
        """Generate loan-specific parameters based on archetype."""

        loan_frequency_map = {
            'regular': np.random.randint(2, 5),      # 2-4 loans in period
            'occasional': np.random.randint(1, 3),   # 1-2 loans
            'frequent': np.random.randint(4, 7),     # 4-6 loans
            'aggressive': np.random.randint(3, 6)    # 3-5 loans (before default)
        }

        repayment_timing_map = {
            'on_time': (0.9, 0.05, 0.05),      # (on_time, late, very_late) probabilities
            'variable': (0.6, 0.3, 0.1),
            'often_late': (0.3, 0.4, 0.3),
            'default': (0.1, 0.2, 0.7)
        }

        freq = archetype_config.get('loan_frequency', 'occasional')
        repay_behavior = archetype_config.get('repayment_behavior', 'variable')

        return {
            'loan_count_target': loan_frequency_map.get(freq, 2),
            'repayment_timing': repayment_timing_map.get(repay_behavior, (0.6, 0.3, 0.1)),
            'default_probability': archetype_config.get('default_probability', 0.05),
            'credit_limit': self.loan_params['initial_limit'],
            'credit_limit_growth': archetype_config.get('credit_limit_growth', True),
            'preferred_loan_provider': np.random.choice(self.LOAN_PROVIDERS),
            'loan_amount_preference': np.random.uniform(0.5, 0.9)  # % of limit typically borrowed
        }

    # ---------------------------------------------------------------
    # Amount generation (type-specific, calibrated to Table 5)
    # ---------------------------------------------------------------

    def generate_transaction_amount(self, user_profile, txn_type):
        """Generate transaction amount based on type and user profile."""

        if txn_type == 'TRANSFER':
            return self._generate_transfer_amount(user_profile)
        elif txn_type == 'PAYMENT_SEND':
            return self._generate_payment_send_amount()
        elif txn_type == 'CASH_OUT':
            return self._generate_cashout_amount()
        elif txn_type == 'CASH_IN':
            return round(np.random.uniform(50, 200), 2)
        else:
            # Fallback for any other type (DEBIT and PAYMENT handled separately)
            base = np.random.lognormal(user_profile['amount_mu'], user_profile['amount_sigma'])
            return round(max(0.5, base), 2)

    def _generate_transfer_amount(self, user_profile):
        """Generate TRANSFER amount - common values end in .0 or .5."""
        if np.random.random() < 0.70:
            bases = [4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 26, 27, 28, 30, 34, 37, 39, 40,
                     50, 51, 60, 70, 75, 80, 82, 86, 101]
            weights = [0.02, 0.03, 0.01, 0.02, 0.04, 0.04, 0.02, 0.02, 0.03, 0.04,
                       0.03, 0.06, 0.06, 0.08, 0.03, 0.02, 0.04, 0.03, 0.08, 0.02,
                       0.03, 0.01, 0.04, 0.02, 0.02, 0.02, 0.04, 0.03, 0.02, 0.02,
                       0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            weights = np.array(weights)
            weights = weights / weights.sum()
            base = np.random.choice(bases, p=weights)
            suffix = np.random.choice([0, 0.5], p=[0.30, 0.70])
            return base + suffix
        else:
            amount = np.random.lognormal(user_profile['amount_mu'], user_profile['amount_sigma'])
            amount = round(amount * 2) / 2  # Round to nearest 0.5
            return max(0.5, amount)

    def _generate_payment_send_amount(self):
        """Generate PAYMENT_SEND amount - concentrated around 18-26 GHS."""
        amounts = [7.5, 10, 12, 12.5, 14.3, 15, 15.5, 16, 18, 18.2, 18.3,
                   18.5, 19, 19.5, 20, 20.5, 21, 21.5, 25, 26, 26.5, 34, 57]
        weights = [0.02, 0.03, 0.03, 0.02, 0.02, 0.04, 0.04, 0.03, 0.05, 0.04, 0.04,
                   0.08, 0.08, 0.08, 0.06, 0.04, 0.06, 0.04, 0.05, 0.05, 0.03, 0.02, 0.02]
        weights = np.array(weights)
        weights = weights / weights.sum()
        return float(np.random.choice(amounts, p=weights))

    def _generate_cashout_amount(self):
        """Generate CASH_OUT amount - standard withdrawal denominations."""
        amounts = [20, 25, 40, 50, 60, 70, 75, 90, 100, 150]
        weights = [0.04, 0.04, 0.12, 0.16, 0.08, 0.08, 0.08, 0.04, 0.24, 0.12]
        return float(np.random.choice(amounts, p=weights))

    def _generate_debit_transaction(self):
        """Generate DEBIT provider, amount, and recipient details."""
        provider_key = np.random.choice(
            self._debit_provider_names, p=self._debit_provider_weights
        )
        provider = self.DEBIT_PROVIDERS[provider_key]

        if provider['amount_type'] == 'fixed':
            amount = provider['fixed_amount']
        elif provider['amount_type'] == 'discrete':
            amount = float(np.random.choice(
                provider['amounts'], p=provider['amount_weights']
            ))
        else:  # range
            amount = round(np.random.uniform(
                provider['amount_min'], provider['amount_max']
            ), 2)

        recipient = {
            'phone': provider['to_no'],
            'name': provider['to_name'],
            'account': provider['to_acct'],
        }

        return provider_key, amount, recipient

    def _generate_payment_transaction(self):
        """Generate PAYMENT sub-type, amount, and recipient details."""
        subtype_key = np.random.choice(
            self._payment_subtype_names, p=self._payment_subtype_weights
        )
        subtype = self.PAYMENT_SUBTYPES[subtype_key]

        if subtype_key == 'airtime':
            amount = float(np.random.choice(
                subtype['amounts'], p=subtype['amount_weights']
            ))
            recipient = {
                'phone': subtype['to_no'],
                'name': subtype['to_name'],
                'account': subtype['to_acct'],
            }
            ova = subtype['ova']
        else:
            # Cross-network payment
            amount = round(np.random.uniform(
                subtype['amount_min'], subtype['amount_max']
            ), 2)
            idx = np.random.randint(len(subtype['to_name_options']))
            recipient = {
                'phone': '0',
                'name': subtype['to_name_options'][idx],
                'account': subtype['to_acct_options'][idx],
            }
            ova = subtype['ova_options'][idx]

        return subtype_key, amount, recipient, ova

    # ---------------------------------------------------------------
    # Fee and E-Levy calculations (calibrated to Table 5)
    # ---------------------------------------------------------------

    def calculate_fees(self, amount, txn_type, provider=None, is_incoming=False):
        """Calculate transaction fees and e-levy based on real patterns."""

        fees = 0.0
        elevy = 0.0

        # Incoming transactions never have fees/e-levy for the receiver
        if is_incoming:
            return fees, elevy

        # Fee calculation by type
        if txn_type == 'CASH_OUT':
            # CASH_OUT: exactly 1% fee, always applied, min 0.50 GHS
            fees = round(max(0.50, amount * 0.01), 2)

        elif txn_type == 'PAYMENT_SEND':
            # PAYMENT_SEND: ~0.5% fee, always applied, min 0.03 GHS
            fees = round(max(0.03, amount * 0.005), 2)

        elif txn_type == 'TRANSFER':
            # TRANSFER: usually 0, ~3.6% chance of fee
            if np.random.random() < self.calibration.get('transfer_fee_probability', 0.036):
                if amount <= 30:
                    fees = 0.38
                else:
                    fees = round(amount * 0.0075, 2)

        elif txn_type == 'PAYMENT' and provider == 'cross_network':
            # Cross-network payments have transfer-like fees
            if amount <= 30:
                fees = 0.38
            else:
                fees = round(amount * 0.0075, 2)

        # E-Levy calculation (~1% rate, no minimum amount threshold)
        if txn_type == 'TRANSFER' and not is_incoming:
            if np.random.random() < 0.18:
                elevy = round(amount * 0.01, 2)
        elif txn_type == 'PAYMENT_SEND':
            if np.random.random() < 0.08:
                elevy = round(amount * 0.01, 2)
        elif txn_type == 'DEBIT' and provider == 'one4all.sp':
            if np.random.random() < 0.10:
                elevy = round(amount * 0.01, 2)
        elif txn_type == 'PAYMENT' and provider == 'cross_network':
            if np.random.random() < 0.15:
                elevy = round(amount * 0.01, 2)

        return fees, elevy

    # ---------------------------------------------------------------
    # Transaction type selection
    # ---------------------------------------------------------------

    def _select_transaction_type(self, user_profile, current_balance):
        """Select transaction type based on user preferences and balance."""

        # If balance is low, might need cash in
        if current_balance < user_profile['min_balance_threshold']:
            if np.random.random() < 0.7:
                return 'CASH_IN'

        types = ['TRANSFER', 'DEBIT', 'PAYMENT_SEND', 'CASH_OUT', 'PAYMENT', 'ADJUSTMENT']
        probs = [
            user_profile['pref_transfer'],
            user_profile['pref_debit'],
            user_profile['pref_payment_send'],
            user_profile['pref_cashout'],
            user_profile['pref_payment'],
            user_profile['pref_adjustment'],
        ]
        probs = np.array(probs) / sum(probs)
        return np.random.choice(types, p=probs)

    def _generate_timestamp(self, current_date, user_profile):
        """Generate transaction timestamp based on user preferences."""

        if np.random.random() < user_profile['pref_night']:
            hour = np.random.choice([22, 23, 0, 1, 2, 3])
        else:
            hour = int(np.clip(
                np.random.normal(user_profile['pref_hour'], 3),
                6, 22
            ))

        return current_date.replace(
            hour=hour,
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60)
        )

    def _select_recipient(self, txn_type, user_recipients):
        """Select transaction recipient for TRANSFER, PAYMENT_SEND, CASH_OUT."""

        if len(user_recipients) > 0 and np.random.random() < 0.6:
            return np.random.choice(user_recipients)
        else:
            recipient = np.random.choice(self.recipient_pool)
            user_recipients.append(recipient)
            return recipient

    # ---------------------------------------------------------------
    # F_ID, REF, OVA generation
    # ---------------------------------------------------------------

    def _generate_f_id(self):
        """Generate an 11-digit numeric transaction ID."""
        return str(np.random.randint(34000000000, 45000000000))

    def _generate_ref(self, txn_type, provider=None, amount=0.0,
                      elevy=0.0, is_incoming=False, incoming_source=None):
        """Generate REF field based on transaction type and provider."""

        if txn_type == 'TRANSFER':
            if is_incoming:
                if incoming_source == 'family':
                    return np.random.choice(self.INCOMING_FAMILY_REFS)
                elif incoming_source == 'bank_to_wallet':
                    return 'Bank To Wallet'
                elif incoming_source == 'interoperability':
                    name = self._generate_name()
                    phone = f'233{np.random.randint(200000000, 600000000)}'
                    network = np.random.choice(['GCB Bank', 'VODAFONE', 'AIRTEL'])
                    return f'{name} ,{phone},GIPS Transfer from {network}'
                else:
                    return np.random.choice(self.INCOMING_FRIEND_REFS)
            else:
                # Outgoing transfer
                if np.random.random() < 0.85:
                    return '1'
                return np.random.choice(['q', 'W', '', '1234'])

        elif txn_type == 'DEBIT':
            rand_id = ''.join([str(np.random.randint(0, 10)) for _ in range(18)])
            if provider == 'one4all.sp':
                return f'One4all Debit SystemIlIELEVYIlI{elevy:.2f}IlI{rand_id}IlI{amount:.1f}'
            elif provider in ('cis', 'cisnew'):
                return np.random.choice(['your Requested', 'Internet Bundle'])
            elif provider == 'HubtelPOS.sp':
                return f'The Cloud NetworkIlIELEVYIlI0.00IlI{rand_id}IlI0.0'
            elif provider == 'J4U':
                return f'GH12_2.2GB Just4UIlIELEVYIlI0.00IlI{rand_id}IlI0.0'
            elif provider == 'Cell.sp':
                alphanum = ''.join(np.random.choice(list(string.ascii_letters + string.digits), size=12))
                return f'JUMIA-{alphanum}IlIELEVYIlI0.00IlI{rand_id}IlI0.0'
            elif provider == 'calpush.sp':
                base_amt = round(amount - (amount % 1), 2) if amount > 0 else 0
                return f'Authorize movement Amt: GHS{base_amt:.2f} to from your walletIlIELEVYIlI0.00IlI{rand_id}IlI0.0'
            elif provider == 'pstack.sp':
                return np.random.choice(['Plus payment', 'Bloom Graphics and Software payment'])
            return ''

        elif txn_type == 'PAYMENT_SEND':
            return np.random.choice(['1', '1', '1', 'q'])

        elif txn_type == 'CASH_OUT':
            return np.random.choice(['NationalId--'] * 7 + [''] * 3)

        elif txn_type == 'PAYMENT':
            if provider == 'airtime':
                return ''
            elif provider == 'cross_network':
                name = self._generate_name()
                phone = f'233{np.random.randint(200000000, 600000000)}'
                return f'{name},{phone},1'
            return ''

        elif txn_type == 'CASH_IN':
            return ''

        elif txn_type == 'ADJUSTMENT':
            return ''

        return ''

    def _generate_ova(self, txn_type, provider=None):
        """Generate OVA (originator) field based on transaction type."""

        if txn_type in ('TRANSFER', 'PAYMENT_SEND', 'CASH_OUT', 'CASH_IN', 'ADJUSTMENT'):
            return 'Internal'
        elif txn_type == 'DEBIT':
            return provider if provider else 'Internal'
        elif txn_type == 'PAYMENT':
            if provider == 'airtime':
                return 'MTNONLINEAIRTIMEVENDO'
            elif provider == 'cross_network':
                return np.random.choice(['mmipush', 'mmipushtigo'])
            return 'Internal'
        return 'Internal'

    # ---------------------------------------------------------------
    # Incoming transaction generation
    # ---------------------------------------------------------------

    def _generate_incoming_transaction(self, user_profile, current_date, current_balance):
        """Generate an incoming TRANSFER transaction to the user."""

        # Determine source category
        categories = []
        cat_weights = []

        categories.append('friends')
        cat_weights.append(0.35)

        if user_profile['has_family_support']:
            categories.append('family')
            cat_weights.append(0.35)
        else:
            cat_weights[0] += 0.15  # friends get more weight
            categories.append('family')
            cat_weights.append(0.20)

        if user_profile['has_bank_link']:
            categories.append('bank_to_wallet')
            cat_weights.append(0.15)
        else:
            categories.append('bank_to_wallet')
            cat_weights.append(0.05)

        categories.append('interoperability')
        cat_weights.append(0.10)

        cat_weights = np.array(cat_weights)
        cat_weights = cat_weights / cat_weights.sum()
        source = np.random.choice(categories, p=cat_weights)

        # Generate sender details and amount based on source
        if source == 'family':
            contact = np.random.choice(user_profile['incoming_contacts'][:2]) \
                if len(user_profile['incoming_contacts']) >= 2 \
                else np.random.choice(user_profile['incoming_contacts'])
            amount = float(np.random.choice(self.INCOMING_FAMILY_AMOUNTS))
            from_acct = contact['account']
            from_name = contact['name']
            from_no = contact['phone']

        elif source == 'friends':
            contact = np.random.choice(user_profile['incoming_contacts'])
            # Friend amounts: 10-100 GHS, common values
            common_amounts = [12, 15, 16, 17, 17.5, 18, 18.5, 20, 20.5, 21,
                              25, 25.5, 30, 31, 40, 40.5, 41, 50, 51, 82, 100]
            amount = float(np.random.choice(common_amounts))
            from_acct = contact['account']
            from_name = contact['name']
            from_no = contact['phone']

        elif source == 'bank_to_wallet':
            amount = float(np.random.choice([20, 30, 50, 100, 250, 270, 350, 500]))
            from_acct = '46831706'
            from_name = 'calpull.sp'
            from_no = '0'

        else:  # interoperability
            amount = round(np.random.uniform(10, 100), 2)
            from_acct = str(np.random.randint(50000000, 75000000))
            from_name = np.random.choice(['INTEROPERABILITY PULL', 'INTEROPERABILITY PULL OVA'])
            from_no = f'233{np.random.randint(240000000, 600000000)}'

        ref = self._generate_ref('TRANSFER', is_incoming=True, incoming_source=source)
        f_id = self._generate_f_id()

        return {
            'TRANSACTION DATE': self._generate_timestamp(current_date, user_profile),
            'FROM ACCT': from_acct,
            'FROM NAME': from_name,
            'FROM NO.': from_no,
            'TRANS. TYPE': 'TRANSFER',
            'AMOUNT': amount,
            'FEES': 0.0,
            'E-LEVY': 0.0,
            'BAL BEFORE': round(current_balance, 2),
            'BAL AFTER': round(current_balance + amount, 2),
            'TO NO.': user_profile['phone_number'],
            'TO NAME': user_profile['user_name'],
            'TO ACCT': user_profile['account_number'],
            'F_ID': f_id,
            'REF': ref,
            'OVA': 'Internal',
        }, amount

    # ---------------------------------------------------------------
    # Loan transactions
    # ---------------------------------------------------------------

    def generate_loan_transaction(self, user_profile, current_date, current_balance,
                                   is_disbursement=True, loan_amount=None):
        """Generate a CREDIT or LOAN_REPAYMENT transaction."""

        provider = user_profile.get('preferred_loan_provider', 'QWIKLOAN')

        if is_disbursement:
            # CREDIT: Loan disbursement
            credit_limit = user_profile.get('credit_limit', self.loan_params['initial_limit'])
            pref = user_profile.get('loan_amount_preference', 0.7)

            # Calculate loan amount (percentage of available limit)
            max_loan = min(credit_limit, self.loan_params['max_amount'])
            min_loan = self.loan_params['min_amount']

            amount = round(np.random.uniform(
                min_loan,
                max_loan * pref
            ), 2)
            amount = max(min_loan, min(amount, max_loan))

            transaction = {
                'TRANSACTION DATE': self._generate_timestamp(current_date, user_profile),
                'FROM ACCT': f'{provider}_LENDING',
                'FROM NAME': f'{provider} Loan Service',
                'FROM NO.': '0',
                'TRANS. TYPE': 'CREDIT',
                'AMOUNT': amount,
                'FEES': 0.0,
                'E-LEVY': 0.0,
                'BAL BEFORE': round(current_balance, 2),
                'BAL AFTER': round(current_balance + amount, 2),
                'TO NO.': user_profile['phone_number'],
                'TO NAME': user_profile['user_name'],
                'TO ACCT': user_profile['account_number'],
                'F_ID': self._generate_f_id(),
                'REF': f'{provider} Loan Disbursement',
                'OVA': 'Internal',
                'LOAN_PROVIDER': provider,
                'LOAN_PRINCIPAL': amount,
                'LOAN_INTEREST_RATE': self.loan_params['interest_rate'],
                'LOAN_DUE_DATE': (current_date + timedelta(days=self.loan_params['term_days'])).strftime('%Y-%m-%d')
            }

            return transaction, amount

        else:
            # LOAN_REPAYMENT: Repaying the loan
            if loan_amount is None:
                return None, 0

            # Calculate repayment amount with interest
            interest = loan_amount * self.loan_params['interest_rate']
            total_due = loan_amount + interest

            # Check if user can afford repayment
            repay_amount = min(total_due, current_balance - 5)  # Keep minimum balance
            repay_amount = max(0, round(repay_amount, 2))

            if repay_amount <= 0:
                return None, 0

            transaction = {
                'TRANSACTION DATE': self._generate_timestamp(current_date, user_profile),
                'FROM ACCT': user_profile['account_number'],
                'FROM NAME': user_profile['user_name'],
                'FROM NO.': user_profile['phone_number'],
                'TRANS. TYPE': 'LOAN_REPAYMENT',
                'AMOUNT': repay_amount,
                'FEES': 0.0,
                'E-LEVY': 0.0,
                'BAL BEFORE': round(current_balance, 2),
                'BAL AFTER': round(current_balance - repay_amount, 2),
                'TO NO.': '0',
                'TO NAME': f'{provider} Loan Service',
                'TO ACCT': f'{provider}_LENDING',
                'F_ID': self._generate_f_id(),
                'REF': f'{provider} Loan Repayment',
                'OVA': 'Internal',
                'LOAN_PROVIDER': provider,
                'LOAN_PRINCIPAL_PAID': round(repay_amount - (repay_amount * self.loan_params['interest_rate'] / (1 + self.loan_params['interest_rate'])), 2),
                'LOAN_INTEREST_PAID': round(repay_amount * self.loan_params['interest_rate'] / (1 + self.loan_params['interest_rate']), 2)
            }

            return transaction, repay_amount

    def _determine_repayment_timing(self, user_profile, loan_date):
        """Determine when the user will repay based on their archetype."""

        timing_probs = user_profile.get('repayment_timing', (0.6, 0.3, 0.1))
        default_prob = user_profile.get('default_probability', 0.05)

        # Check for default
        if np.random.random() < default_prob:
            return None, 2  # Default - no repayment, risk label 2

        timing_choice = np.random.choice(['on_time', 'late', 'very_late'], p=timing_probs)

        if timing_choice == 'on_time':
            days = np.random.randint(7, 30)
            risk_label = 0
        elif timing_choice == 'late':
            days = np.random.randint(31, 45)
            risk_label = 1
        else:  # very_late
            days = np.random.randint(45, 60)
            risk_label = 1

        repay_date = loan_date + timedelta(days=days)
        return repay_date, risk_label

    # ---------------------------------------------------------------
    # Main transaction generation loop
    # ---------------------------------------------------------------

    def generate_transactions_for_user(self, user_profile):
        """Generate complete transaction sequence for a single user."""

        transactions = []
        current_date = self.start_date + timedelta(days=user_profile['account_start_day'])
        current_balance = user_profile['initial_balance']

        # Tracking
        user_recipients = []
        n_transactions = max(5, min(50, int(np.random.poisson(self.avg_transactions_per_user))))

        # Loan tracking
        active_loan = None
        loans_taken = 0
        loan_target = user_profile.get('loan_count_target', 0) if user_profile['takes_loans'] else 0
        credit_risk_labels = []

        # Calculate loan timing based on transaction index
        loan_at_txn_indices = []
        if loan_target > 0:
            total_expected = n_transactions
            first_loan_idx = max(3, int(total_expected * 0.15))
            loan_interval = max(3, (total_expected - first_loan_idx) // (loan_target + 1))

            for i in range(loan_target):
                loan_idx = first_loan_idx + (i * loan_interval) + np.random.randint(-1, 2)
                loan_idx = max(3, min(loan_idx, total_expected - 5))
                loan_at_txn_indices.append(loan_idx)

            loan_at_txn_indices.sort()

        incoming_rate = self.calibration.get('incoming_rate', 0.28)

        txn_idx = 0
        regular_txn_count = 0
        while current_date < self.end_date and regular_txn_count < n_transactions:
            txn_idx += 1

            # Check if it's time for a loan (based on transaction count)
            if loan_at_txn_indices and regular_txn_count >= loan_at_txn_indices[0] and active_loan is None:
                loan_at_txn_indices.pop(0)

                # Generate loan disbursement (CREDIT)
                loan_txn, loan_amount = self.generate_loan_transaction(
                    user_profile, current_date, current_balance, is_disbursement=True
                )

                if loan_txn and loan_amount > 0:
                    transactions.append(loan_txn)
                    current_balance += loan_amount
                    loans_taken += 1

                    # Determine repayment timing
                    repay_timing, risk_label = self._determine_repayment_timing(
                        user_profile, current_date
                    )

                    if repay_timing is None:
                        repay_at_txn = None  # Default - no repayment
                    else:
                        remaining_txns = n_transactions - regular_txn_count
                        if remaining_txns > 3:
                            txns_to_repay = np.random.randint(2, min(7, remaining_txns))
                            repay_at_txn = regular_txn_count + txns_to_repay
                        else:
                            repay_at_txn = regular_txn_count + max(1, remaining_txns - 1)

                    active_loan = {
                        'amount': loan_amount,
                        'disbursement_date': current_date,
                        'repay_at_txn': repay_at_txn,
                        'risk_label': risk_label
                    }
                    credit_risk_labels.append(risk_label)

                    # Grow credit limit if successful borrower
                    if user_profile.get('credit_limit_growth', False) and risk_label == 0:
                        current_limit = user_profile.get('credit_limit', self.loan_params['initial_limit'])
                        new_limit = min(
                            current_limit * self.loan_params['limit_growth_rate'],
                            self.loan_params['max_limit']
                        )
                        user_profile['credit_limit'] = new_limit

            # Check if it's time to repay loan (based on transaction count)
            if active_loan and active_loan.get('repay_at_txn') is not None:
                if regular_txn_count >= active_loan['repay_at_txn']:
                    repay_txn, repay_amount = self.generate_loan_transaction(
                        user_profile, current_date, current_balance,
                        is_disbursement=False, loan_amount=active_loan['amount']
                    )

                    if repay_txn and repay_amount > 0:
                        transactions.append(repay_txn)
                        current_balance -= repay_amount

                    active_loan = None

            # Select transaction type
            txn_type = self._select_transaction_type(user_profile, current_balance)

            # --- Handle incoming TRANSFER ---
            if txn_type == 'TRANSFER' and np.random.random() < incoming_rate:
                txn_data, inc_amount = self._generate_incoming_transaction(
                    user_profile, current_date, current_balance
                )
                current_balance += inc_amount
                txn_data['BAL BEFORE'] = round(current_balance - inc_amount, 2)
                txn_data['BAL AFTER'] = round(current_balance, 2)
                transactions.append(txn_data)
                regular_txn_count += 1

                # Advance time
                hours_gap = np.random.exponential(user_profile['hours_between_txns'])
                if current_date.weekday() >= 5 and np.random.random() < user_profile['pref_weekend']:
                    hours_gap *= 0.7
                current_date += timedelta(hours=max(0.5, hours_gap))
                continue

            # --- Handle ADJUSTMENT (interest credit) ---
            if txn_type == 'ADJUSTMENT':
                adj_amount = round(np.random.uniform(1.0, 6.0), 2)
                balance_before = current_balance
                current_balance += adj_amount

                transaction = {
                    'TRANSACTION DATE': self._generate_timestamp(current_date, user_profile),
                    'FROM ACCT': '48878389',
                    'FROM NAME': 'momointerest.sp',
                    'FROM NO.': '0',
                    'TRANS. TYPE': 'ADJUSTMENT',
                    'AMOUNT': adj_amount,
                    'FEES': 0.0,
                    'E-LEVY': 0.0,
                    'BAL BEFORE': round(balance_before, 2),
                    'BAL AFTER': round(current_balance, 2),
                    'TO NO.': user_profile['phone_number'],
                    'TO NAME': user_profile['user_name'],
                    'TO ACCT': user_profile['account_number'],
                    'F_ID': self._generate_f_id(),
                    'REF': '',
                    'OVA': 'Internal',
                }
                transactions.append(transaction)
                regular_txn_count += 1

                hours_gap = np.random.exponential(user_profile['hours_between_txns'])
                if current_date.weekday() >= 5 and np.random.random() < user_profile['pref_weekend']:
                    hours_gap *= 0.7
                current_date += timedelta(hours=max(0.5, hours_gap))
                continue

            # --- Handle DEBIT (provider-specific) ---
            provider_key = None
            if txn_type == 'DEBIT':
                provider_key, amount, recipient = self._generate_debit_transaction()

            # --- Handle PAYMENT (airtime vs cross-network) ---
            elif txn_type == 'PAYMENT':
                provider_key, amount, recipient, _ = self._generate_payment_transaction()

            # --- Handle TRANSFER, PAYMENT_SEND, CASH_OUT, CASH_IN ---
            else:
                amount = self.generate_transaction_amount(user_profile, txn_type)
                recipient = self._select_recipient(txn_type, user_recipients)
                if len(user_recipients) > user_profile['typical_recipients']:
                    user_recipients.pop(0)

            # Ensure sufficient balance for outgoing transactions
            if txn_type not in ['CASH_IN']:
                available = current_balance - user_profile['min_balance_threshold']
                if available <= 0:
                    txn_type = 'CASH_IN'
                    amount = np.random.uniform(50, 200)
                    provider_key = None
                    recipient = self._select_recipient(txn_type, user_recipients)
                else:
                    amount = min(amount, available)

            # Calculate fees
            fees, elevy = self.calculate_fees(amount, txn_type, provider=provider_key)

            # Update balance
            balance_before = current_balance
            if txn_type == 'CASH_IN':
                current_balance += amount
            else:
                current_balance -= (amount + fees + elevy)

            # Generate OVA based on type/provider
            ova = self._generate_ova(txn_type, provider=provider_key)

            # For PAYMENT, override OVA from payment sub-type
            if txn_type == 'PAYMENT' and provider_key:
                if provider_key == 'airtime':
                    ova = 'MTNONLINEAIRTIMEVENDO'
                elif provider_key == 'cross_network':
                    ova = np.random.choice(['mmipush', 'mmipushtigo'])

            # Generate REF
            ref = self._generate_ref(
                txn_type, provider=provider_key, amount=amount, elevy=elevy
            )

            # Build transaction
            if txn_type == 'CASH_IN':
                # CASH_IN: sender is the external source, receiver is the user
                from_acct = recipient['account']
                from_name = recipient['name']
                from_no = recipient['phone']
                to_no = user_profile['phone_number']
                to_name = user_profile['user_name']
                to_acct = user_profile['account_number']
            else:
                # Regular outgoing transaction
                from_acct = user_profile['account_number']
                from_name = user_profile['user_name']
                from_no = user_profile['phone_number']
                to_no = recipient['phone']
                to_name = recipient['name']
                to_acct = recipient['account']

            transaction = {
                'TRANSACTION DATE': self._generate_timestamp(current_date, user_profile),
                'FROM ACCT': from_acct,
                'FROM NAME': from_name,
                'FROM NO.': from_no,
                'TRANS. TYPE': txn_type,
                'AMOUNT': amount,
                'FEES': fees,
                'E-LEVY': elevy,
                'BAL BEFORE': round(balance_before, 2),
                'BAL AFTER': round(current_balance, 2),
                'TO NO.': to_no,
                'TO NAME': to_name,
                'TO ACCT': to_acct,
                'F_ID': self._generate_f_id(),
                'REF': ref,
                'OVA': ova,
            }

            transactions.append(transaction)
            regular_txn_count += 1

            # Next transaction time
            hours_gap = np.random.exponential(user_profile['hours_between_txns'])
            if current_date.weekday() >= 5 and np.random.random() < user_profile['pref_weekend']:
                hours_gap *= 0.7

            current_date += timedelta(hours=max(0.5, hours_gap))

        # Handle defaulted loan (no repayment transaction)
        if active_loan and active_loan.get('repay_at_txn') is None:
            pass

        # Sort transactions by date
        transactions.sort(key=lambda x: x['TRANSACTION DATE'])

        # Determine final credit risk label for user
        if credit_risk_labels:
            final_risk_label = max(credit_risk_labels)  # Worst outcome
        else:
            final_risk_label = -1  # No loans taken

        return transactions, {
            'user_id': user_profile['user_id'],
            'credit_archetype': user_profile['credit_archetype'],
            'loans_taken': loans_taken,
            'credit_risk_label': final_risk_label,
            'final_credit_limit': user_profile.get('credit_limit', 0),
            'total_transactions': len(transactions)
        }

    def save_user_transactions(self, user_id, transactions):
        """Save transactions for a single user to CSV."""

        if not transactions:
            return None

        df = pd.DataFrame(transactions)

        # Format date to match real data: DD-MMM-YYYY HH:MM:SS AM/PM
        df['TRANSACTION DATE'] = pd.to_datetime(df['TRANSACTION DATE'])
        df['TRANSACTION DATE'] = df['TRANSACTION DATE'].dt.strftime('%d-%b-%Y %I:%M:%S %p')

        # Ensure column order matches real data format
        base_columns = [
            'TRANSACTION DATE', 'FROM ACCT', 'FROM NAME', 'FROM NO.',
            'TRANS. TYPE', 'AMOUNT', 'FEES', 'E-LEVY',
            'BAL BEFORE', 'BAL AFTER', 'TO NO.', 'TO NAME', 'TO ACCT',
            'F_ID', 'REF', 'OVA'
        ]
        extra_cols = [c for c in df.columns if c not in base_columns]
        column_order = [c for c in base_columns if c in df.columns] + extra_cols
        df = df[column_order]

        filename = f"{user_id}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        return filepath

    def generate_dataset(self):
        """Generate complete dataset with individual files per user."""

        print(f"\n{'='*80}")
        print("GENERATING {0:,} INDIVIDUAL USER DATASETS".format(self.n_users))
        print(f"{'='*80}")

        user_summaries = []
        total_transactions = 0
        archetype_counts = {k: 0 for k in self.CREDIT_ARCHETYPES.keys()}
        risk_label_counts = {-1: 0, 0: 0, 1: 0, 2: 0}

        for i in range(self.n_users):
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i+1:,}/{self.n_users:,} users generated...")

            # Generate user profile
            user_profile = self.generate_user_profile(i)
            archetype_counts[user_profile['credit_archetype']] += 1

            # Generate transactions
            transactions, summary = self.generate_transactions_for_user(user_profile)

            # Save to individual file
            self.save_user_transactions(user_profile['user_id'], transactions)

            # Track statistics
            user_summaries.append(summary)
            total_transactions += len(transactions)
            risk_label_counts[summary['credit_risk_label']] += 1

        # Save user summaries
        summary_df = pd.DataFrame(user_summaries)
        summary_path = self.output_dir.parent / 'user_summaries.csv'
        summary_df.to_csv(summary_path, index=False)

        # Print statistics
        print(f"\n{'='*80}")
        print("GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nTotal users: {self.n_users:,}")
        print(f"Total transactions: {total_transactions:,}")
        print(f"Average transactions per user: {total_transactions / self.n_users:.1f}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"User summaries: {summary_path}")

        print(f"\n{'='*80}")
        print("CREDIT BEHAVIOR DISTRIBUTION")
        print(f"{'='*80}")
        for archetype, count in archetype_counts.items():
            pct = count / self.n_users * 100
            print(f"  {archetype}: {count:,} ({pct:.1f}%)")

        print(f"\n{'='*80}")
        print("CREDIT RISK LABEL DISTRIBUTION")
        print(f"{'='*80}")
        risk_labels = {
            -1: "No loans taken",
            0: "Good (repaid on time)",
            1: "Late (repaid after term)",
            2: "Default (failed to repay)"
        }
        for label, count in risk_label_counts.items():
            pct = count / self.n_users * 100
            print(f"  {label} - {risk_labels[label]}: {count:,} ({pct:.1f}%)")

        # Calculate borrower-specific default rate
        borrowers = self.n_users - risk_label_counts[-1]
        if borrowers > 0:
            defaults = risk_label_counts[2]
            default_rate = defaults / borrowers * 100
            print(f"\nDefault rate (among borrowers): {defaults:,}/{borrowers:,} ({default_rate:.2f}%)")

        return summary_df


def main():
    """Generate calibrated synthetic dataset with per-user files."""

    print("=" * 80)
    print("CALIBRATED MOBILE MONEY DATA GENERATOR v3.0")
    print("Per-User Transaction Datasets with Credit/Loan Functionality")
    print("=" * 80)

    # Initialize generator
    generator = CalibratedMoMoDataGenerator(
        n_users=10000,
        avg_transactions_per_user=15,
        start_date='2024-01-01',
        duration_days=180,
        output_dir='data/user_transactions'
    )

    # Generate dataset
    summary_df = generator.generate_dataset()

    print(f"\n{'='*80}")
    print("SAMPLE USER SUMMARIES")
    print(f"{'='*80}")
    print(summary_df.head(20).to_string())

    return summary_df


if __name__ == "__main__":
    summary_df = main()
