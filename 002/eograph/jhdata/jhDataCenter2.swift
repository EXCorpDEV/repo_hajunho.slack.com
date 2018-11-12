//
//  jhDataCenter.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

protocol s {
    var x: CGFloat { get set }
    var y: CGFloat { get set }
}

/// x-axiS, y-axiS
struct ss : s {
    var x: CGFloat
    var y: CGFloat
}

/// start x-axiS, end x-axiS, start y-axiS, end y-axiS
struct ssss : s {
    var x: CGFloat
    var y: CGFloat
    var x2: CGFloat
    var y2: CGFloat
}

/// grapH jh
public struct hjh2 {
    var d : Array<s>
}

open class jhDataCenter2 {
    
    static var mDatas : [Int:hjh2] = [0:hjh2(d: Array<ss>()),
                                            1:hjh2(d: Array<ssss>()),
                                            2:hjh2(d: Array<ssss>())
    ]
    
    static var mCountOfaxes_view : Int = 1
    static var mCountOfdatas_view : Int = 1
    
    init() {
    }
    
    public static var nonNetworkData : Array<CGFloat> = Array()
    
    public static func convertArrayToSS(src: Array<CGFloat>) -> hjh2 {
        
        var temp = ss.init(x: 0, y: 0)
        var ret = hjh2.init(d: [])
        
        for l in 0..<src.count {
            temp.x = CGFloat(l)
            temp.y = src[l]
            ret.d.append(temp)
        }
        return ret
    }
    
    //    public static var mDatas : Array<Array<CGFloat>> = Array()
    
    private static var listeners = [observer_p]()
    
    static func attachObserver(observer : observer_p) {
        listeners.append(observer)
    }
    
    public static func notiDataDowloadFinish() {
        for x in listeners {
            x.jhRedraw()
        }
    }
    
    //    public func getData() -> Array<CGFloat> {
    //        return self.mValuesOfDatas
    //    }
    //
    //    public func setData(x: Array<CGFloat>) {
    //        self.mValuesOfDatas = x
    //    }
}
