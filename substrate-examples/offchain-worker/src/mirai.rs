

pub mod mirai_check {
    use frame_support::pallet_prelude::*;
    use crate::Pallet;
    use crate::tests::Test;

    pub fn code_to_analyze(block_number: u64, price: u32) {
        let call: crate::Call<Test> = crate::Call::submit_price_unsigned {
            block_number: block_number,
            price: price,
        };

        let validity = Pallet::validate_unsigned(TransactionSource::Local, &call).unwrap_err();
    }
}