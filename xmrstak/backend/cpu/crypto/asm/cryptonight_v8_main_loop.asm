_TEXT_CNV8_MAINLOOP SEGMENT PAGE READ EXECUTE
PUBLIC cryptonight_v8_mainloop_ivybridge_asm
PUBLIC cryptonight_v8_mainloop_ryzen_asm

ALIGN 8
cryptonight_v8_mainloop_ivybridge_asm PROC
        INCLUDE cryptonight_v8_main_loop_ivybridge_win64.inc
	ret 0
cryptonight_v8_mainloop_ivybridge_asm ENDP

ALIGN 8
cryptonight_v8_mainloop_ryzen_asm PROC
        INCLUDE cryptonight_v8_main_loop_ryzen_win64.inc
	ret 0
cryptonight_v8_mainloop_ryzen_asm ENDP

_TEXT_CNV8_MAINLOOP ENDS
END
