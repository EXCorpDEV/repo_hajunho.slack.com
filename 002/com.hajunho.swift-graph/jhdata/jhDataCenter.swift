//
//  jhData.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 19..
//  Copyright © 2018년 hajunho.com. All rights reserved.
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
struct hjh {
    var d : Array<s>
}

class jhDataCenter {
    
    public static var mDatas : [Int:hjh] = [0:hjh(d: Array<s>()),
                                            1:hjh(d: Array<s>()),
                                            2:hjh(d: Array<s>())
    ]
    
    public static var mCountOfaxes_view : Int = 1
    
    init() {
    }
    
    public static var nonNetworkData : Array<CGFloat> = Array()
    
    public static func convertArrayToSS(src: Array<CGFloat>) -> hjh {
        
        var temp = ss.init(x: 0, y: 0)
        var ret = hjh.init(d: [])
        
        for l in 0..<src.count {
            temp.x = CGFloat(l)
            temp.y = src[l]
            ret.d.append(temp)
        }
        return ret
    }
    
    public static var mValuesOfDatas2 : Array<CGFloat> = Array() {
        didSet {
            if GS.shared.logLevel.contains(.graph) {
                print("mValuesOfDatas.count has been changed to \(mValuesOfDatas2.count) in jhPanel")
            }
        }
    }
    
    public static var mValuesOfDatas3 : Array<CGFloat> = Array() {
        didSet {
            if GS.shared.logLevel.contains(.graph) {
                print("mValuesOfDatas.count has been changed to \(mValuesOfDatas3.count) in jhPanel")
            }
        }
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
