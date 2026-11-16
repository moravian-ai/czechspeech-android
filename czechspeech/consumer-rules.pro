# CzechSpeech: the JNI library binds to these names (Java_ai_moravian_czechspeech_internal_NativeBridge_*).
-keep,includedescriptorclasses class ai.moravian.czechspeech.internal.NativeBridge {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class ai.moravian.czechspeech.** {
    native <methods>;
}
