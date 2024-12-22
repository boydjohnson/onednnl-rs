use {
    crate::error::DnnlError,
    onednnl_sys::{
        dnnl_accumulation_mode_t, dnnl_primitive_attr_create, dnnl_primitive_attr_destroy,
        dnnl_primitive_attr_get_accumulation_mode, dnnl_primitive_attr_get_deterministic,
        dnnl_primitive_attr_set_accumulation_mode, dnnl_primitive_attr_set_deterministic,
        dnnl_primitive_attr_t,
        dnnl_status_t::{self},
    },
};

pub struct PrimitiveAttributes {
    pub(crate) handle: dnnl_primitive_attr_t,
}

impl PrimitiveAttributes {
    pub fn new() -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { dnnl_primitive_attr_create(&mut handle) };

        if status == dnnl_status_t::dnnl_success {
            Ok(Self { handle })
        } else {
            Err(status.into())
        }
    }

    /// Get the accumulation mode
    ///
    /// ```
    /// use onednnl::primitive::attributes::PrimitiveAttributes;
    /// use onednnl_sys::dnnl_accumulation_mode_t;
    ///
    /// let attr = PrimitiveAttributes::new();
    ///
    /// assert!(attr.is_ok());
    ///
    /// let attr = attr.unwrap();
    ///
    /// assert_eq!(attr.get_accumulation_mode(), Ok(dnnl_accumulation_mode_t::dnnl_accumulation_mode_strict));
    ///
    /// ```
    pub fn get_accumulation_mode(&self) -> Result<dnnl_accumulation_mode_t::Type, DnnlError> {
        let mut output = 0;
        let status = unsafe { dnnl_primitive_attr_get_accumulation_mode(self.handle, &mut output) };

        if status == dnnl_status_t::dnnl_success {
            Ok(output)
        } else {
            Err(status.into())
        }
    }

    /// Set the accumulation mode
    ///
    /// ```
    /// use onednnl::primitive::attributes::PrimitiveAttributes;
    /// use onednnl_sys::dnnl_accumulation_mode_t;
    ///
    /// let attr = PrimitiveAttributes::new();
    ///
    /// assert!(attr.is_ok());
    ///
    /// let mut attr = attr.unwrap();
    ///
    /// assert!(attr.set_accumulation_mode(dnnl_accumulation_mode_t::dnnl_accumulation_mode_any).is_ok());
    ///
    /// assert_eq!(attr.get_accumulation_mode(), Ok(dnnl_accumulation_mode_t::dnnl_accumulation_mode_any));
    ///
    /// ```
    pub fn set_accumulation_mode(
        &mut self,
        mode: dnnl_accumulation_mode_t::Type,
    ) -> Result<(), DnnlError> {
        let status = unsafe { dnnl_primitive_attr_set_accumulation_mode(self.handle, mode) };

        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }

    /// Get the deterministic attr
    ///
    /// ```
    /// use onednnl::primitive::attributes::PrimitiveAttributes;
    /// use onednnl_sys::dnnl_accumulation_mode_t;
    ///
    /// let attr = PrimitiveAttributes::new();
    ///
    /// assert!(attr.is_ok());
    ///
    /// let attr = attr.unwrap();
    ///
    /// assert_eq!(attr.get_deterministic(), Ok(false));
    ///
    /// ```
    pub fn get_deterministic(&self) -> Result<bool, DnnlError> {
        let mut output = 0;

        let status = unsafe { dnnl_primitive_attr_get_deterministic(self.handle, &mut output) };

        if status == dnnl_status_t::dnnl_success {
            Ok(output == 1)
        } else {
            Err(status.into())
        }
    }

    /// Set the deterministic attr
    ///
    /// ```
    /// use onednnl::primitive::attributes::PrimitiveAttributes;
    /// use onednnl_sys::dnnl_accumulation_mode_t;
    ///
    /// let attr = PrimitiveAttributes::new();
    ///
    /// assert!(attr.is_ok());
    ///
    /// let mut attr = attr.unwrap();
    ///
    /// assert_eq!(attr.get_deterministic(), Ok(false));
    ///
    /// assert_eq!(attr.set_deterministic(true), Ok(()));
    ///
    /// assert_eq!(attr.get_deterministic(), Ok(true));
    /// ```
    pub fn set_deterministic(&mut self, deterministic: bool) -> Result<(), DnnlError> {
        let value = if deterministic { 1 } else { 0 };

        let status = unsafe { dnnl_primitive_attr_set_deterministic(self.handle, value) };
        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }
}

impl Drop for PrimitiveAttributes {
    fn drop(&mut self) {
        unsafe {
            dnnl_primitive_attr_destroy(self.handle);
        }
    }
}
